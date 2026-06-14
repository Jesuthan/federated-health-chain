#!/usr/bin/env node
'use strict';

/**
 * BFL Healthcare Launcher
 * =======================
 * Single entry point — run this ONE command, then control everything from the browser.
 *
 *   node launcher.js
 *   open http://localhost:4000
 *
 * This server can start/stop:
 *   - IPFS daemon
 *   - Hyperledger Fabric test-network + chaincode deploy
 *   - FL REST server (server/server.js)
 *
 * It also proxies all /api/* calls to the FL server once it is running,
 * so the dashboard and simulator work from the same origin (port 4000).
 */

const express    = require('express');
const { spawn }  = require('child_process');
const http       = require('http');
const path       = require('path');
const fs         = require('fs');
const os         = require('os');

const app  = express();
const PORT = process.env.LAUNCHER_PORT || 4000;
const FL_SERVER_PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Process registry ──────────────────────────────────────────────────────────

const procs = {
  ipfs:   { proc: null, logs: [], label: 'IPFS Daemon'         },
  fabric: { proc: null, logs: [], label: 'Fabric Network'      },
  server: { proc: null, logs: [], label: 'FL REST Server'      },
};

// SSE clients per service
const sseClients = { ipfs: [], fabric: [], server: [] };

function pushLog(service, line, level = 'info') {
  const entry = { ts: new Date().toISOString(), line, level };
  const store  = procs[service].logs;
  store.push(entry);
  if (store.length > 300) store.shift();
  sseClients[service].forEach(res => {
    res.write(`data: ${JSON.stringify(entry)}\n\n`);
  });
}

function isRunning(service) {
  const p = procs[service].proc;
  return p !== null && p.exitCode === null;
}

// ── Service launchers ─────────────────────────────────────────────────────────

function startIPFS() {
  if (isRunning('ipfs')) return { ok: false, error: 'IPFS already running' };

  // Look for ipfs binary
  const candidates = [
    'D:\\tmp\\kubo\\kubo\\ipfs.exe',
    'C:\\kubo\\kubo\\ipfs.exe',
    'C:\\Program Files\\kubo\\ipfs.exe',
    'ipfs',
    'ipfs.exe',
  ];
  const bin = candidates.find(c => {
    try { return c.includes('\\') ? fs.existsSync(c) : true; } catch { return false; }
  }) || 'ipfs';

  pushLog('ipfs', `Starting IPFS daemon (${bin})…`, 'info');

  const proc = spawn(bin, ['daemon'], { shell: false });
  procs.ipfs.proc = proc;

  proc.stdout.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('ipfs', l)));
  proc.stderr.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('ipfs', l, 'warn')));
  proc.on('close', code => {
    pushLog('ipfs', `IPFS daemon exited (code ${code})`, code === 0 ? 'info' : 'error');
    procs.ipfs.proc = null;
  });

  return { ok: true };
}

function stopIPFS() {
  const p = procs.ipfs.proc;
  if (!p) return { ok: false, error: 'IPFS not running' };
  p.kill();
  pushLog('ipfs', 'IPFS daemon stopped.', 'warn');
  return { ok: true };
}

// WSL2 Ubuntu is the correct environment for Fabric test-network scripts.
// Git Bash has unresolvable path-conversion conflicts between Docker and Windows binaries.
function wslRun(script) {
  return spawn('wsl', ['-d', 'Ubuntu', '--', 'bash', '-c', script], { shell: false });
}

function fabricSamplesExists() {
  try {
    const r = require('child_process').spawnSync(
      'wsl', ['-d', 'Ubuntu', '--', 'bash', '-c',
        'test -d ~/fabric-samples/test-network && echo yes || echo no']
    );
    return r.stdout.toString().trim() === 'yes';
  } catch { return false; }
}

function startFabric() {
  if (isRunning('fabric')) return { ok: false, error: 'Fabric already running' };

  // Chaincode lives on Windows at D:\tmp\fedlearn-fabric\chaincode\modelregistry
  // From WSL2 that path is /mnt/d/tmp/fedlearn-fabric/chaincode/modelregistry
  const chaincodePath = path.join(__dirname, 'chaincode', 'modelregistry')
    .replace(/\\/g, '/')
    .replace(/^([A-Z]):/, (_, d) => `/mnt/${d.toLowerCase()}`);

  const installStep = fabricSamplesExists()
    ? ''
    : 'cd ~ && curl -sSL https://bit.ly/2ysbOFE | bash -s -- 2.5.0 1.5.5 && ';

  const script = [
    installStep + 'cd ~/fabric-samples/test-network',
    './network.sh down',
    './network.sh up createChannel -ca',
    `./network.sh deployCC -ccn modelregistry -ccp "${chaincodePath}" -ccl javascript`,
  ].join(' && ');

  if (!fabricSamplesExists()) {
    pushLog('fabric', 'fabric-samples not found in WSL2 — installing (this takes a few minutes)…', 'warn');
  }
  pushLog('fabric', 'Starting Fabric network via WSL2 Ubuntu…', 'info');

  const proc = wslRun(script);
  procs.fabric.proc = proc;

  proc.stdout.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('fabric', l)));
  proc.stderr.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('fabric', l, 'warn')));
  proc.on('close', code => {
    pushLog('fabric', `Fabric setup ${code === 0 ? 'complete ✓' : 'failed'} (code ${code})`, code === 0 ? 'info' : 'error');
    procs.fabric.proc = null;
  });

  return { ok: true };
}

function stopFabric() {
  pushLog('fabric', 'Stopping Fabric network…', 'warn');

  wslRun('cd ~/fabric-samples/test-network && ./network.sh down')
    .stdout.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('fabric', l)));

  if (procs.fabric.proc) { procs.fabric.proc.kill(); procs.fabric.proc = null; }
  return { ok: true };
}

function startFLServer() {
  if (isRunning('server')) return { ok: false, error: 'FL server already running' };

  const serverDir    = path.join(__dirname, 'server');
  const serverScript = path.join(serverDir, 'server.js');
  const nodeModules  = path.join(serverDir, 'node_modules');
  const minClients   = process.env.MIN_CLIENTS || '3';

  // Auto-install server deps if missing
  if (!fs.existsSync(nodeModules)) {
    pushLog('server', 'node_modules not found — running npm install in server/…', 'warn');
    const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    const install = require('child_process').spawnSync(npmCmd, ['install'], {
      cwd: serverDir, shell: false, stdio: 'pipe',
    });
    if (install.status !== 0) {
      const err = install.stderr.toString();
      pushLog('server', `npm install failed: ${err}`, 'error');
      return { ok: false, error: 'npm install failed' };
    }
    pushLog('server', 'npm install complete.', 'info');
  }

  pushLog('server', `Starting FL REST server on port ${FL_SERVER_PORT}…`, 'info');

  const proc = spawn(process.execPath, [serverScript], {
    cwd: serverDir,
    env: { ...process.env, PORT: String(FL_SERVER_PORT), MIN_CLIENTS: minClients },
    shell: false,
  });
  procs.server.proc = proc;

  proc.stdout.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('server', l)));
  proc.stderr.on('data', d => d.toString().split('\n').filter(Boolean).forEach(l => pushLog('server', l, 'error')));
  proc.on('close', code => {
    pushLog('server', `FL server exited (code ${code})`, code === 0 ? 'info' : 'error');
    procs.server.proc = null;
  });

  return { ok: true };
}

function stopFLServer() {
  const p = procs.server.proc;
  if (!p) return { ok: false, error: 'FL server not running' };
  p.kill();
  pushLog('server', 'FL server stopped.', 'warn');
  return { ok: true };
}

function runEnrollAdmin(res) {
  const serverDir = path.join(__dirname, 'server');
  pushLog('server', 'Running enrollAdmin.js…', 'info');

  const enroll = require('child_process').spawnSync(
    process.execPath, ['enrollAdmin.js'],
    { cwd: serverDir, shell: false, stdio: 'pipe', env: { ...process.env } }
  );
  (enroll.stdout.toString() + enroll.stderr.toString()).split('\n').filter(Boolean).forEach(l => pushLog('server', l));

  if (enroll.status !== 0) {
    const err = 'enrollAdmin failed — is the Fabric network running?';
    pushLog('server', err, 'error');
    return res.status(500).json({ ok: false, error: err });
  }

  pushLog('server', 'Admin enrolled successfully.', 'info');
  return res.json({ ok: true, message: 'Admin enrolled' });
}

function runRegisterUser(res) {
  const serverDir = path.join(__dirname, 'server');
  pushLog('server', 'Running registerUser.js…', 'info');

  const register = require('child_process').spawnSync(
    process.execPath, ['registerUser.js'],
    { cwd: serverDir, shell: false, stdio: 'pipe', env: { ...process.env } }
  );
  (register.stdout.toString() + register.stderr.toString()).split('\n').filter(Boolean).forEach(l => pushLog('server', l));

  if (register.status !== 0) {
    const err = 'registerUser failed — check FL Server log for details';
    pushLog('server', err, 'error');
    return res.status(500).json({ ok: false, error: err });
  }

  pushLog('server', 'appUser registered and enrolled. Wallet ready.', 'info');
  return res.json({ ok: true, message: 'appUser registered and enrolled' });
}

// ── Routes ────────────────────────────────────────────────────────────────────

app.get('/launcher/status', (_req, res) => {
  res.json({
    ipfs:   isRunning('ipfs'),
    fabric: isRunning('fabric'),
    server: isRunning('server'),
  });
});

app.post('/launcher/:service/start', (req, res) => {
  const { service } = req.params;
  const result =
    service === 'ipfs'   ? startIPFS()    :
    service === 'fabric' ? startFabric()  :
    service === 'server' ? startFLServer():
    { ok: false, error: 'Unknown service' };
  res.json(result);
});

app.post('/launcher/:service/stop', (req, res) => {
  const { service } = req.params;
  const result =
    service === 'ipfs'   ? stopIPFS()   :
    service === 'fabric' ? stopFabric() :
    service === 'server' ? stopFLServer():
    { ok: false, error: 'Unknown service' };
  res.json(result);
});

// Wallet setup — two separate steps
app.post('/launcher/server/enroll-admin',   (req, res) => runEnrollAdmin(res));
app.post('/launcher/server/register-user',  (req, res) => runRegisterUser(res));

// Wallet status — check which identities exist in the wallet
app.get('/launcher/server/wallet-status', (_req, res) => {
  const walletPath = path.join(__dirname, 'server', 'wallet');
  const has = (name) => {
    try { return fs.existsSync(path.join(walletPath, `${name}.id`)); } catch { return false; }
  };
  res.json({ admin: has('admin'), appUser: has('appUser') });
});

// SSE log stream
app.get('/launcher/:service/logs', (req, res) => {
  const { service } = req.params;
  if (!procs[service]) return res.status(404).end();

  res.setHeader('Content-Type',  'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection',    'keep-alive');
  res.flushHeaders();

  // Send history
  procs[service].logs.forEach(e => res.write(`data: ${JSON.stringify(e)}\n\n`));

  sseClients[service].push(res);
  req.on('close', () => {
    const idx = sseClients[service].indexOf(res);
    if (idx !== -1) sseClients[service].splice(idx, 1);
  });
});

// ── Proxy /api/* -> FL server ─────────────────────────────────────────────────

app.use('/api', (req, res) => {
  const options = {
    hostname: 'localhost',
    port:     FL_SERVER_PORT,
    path:     `/api${req.url}`,
    method:   req.method,
    headers:  { ...req.headers, host: `localhost:${FL_SERVER_PORT}` },
  };

  const proxy = http.request(options, (upstream) => {
    res.writeHead(upstream.statusCode, upstream.headers);
    upstream.pipe(res);
  });

  proxy.on('error', () => {
    if (!res.headersSent) {
      res.status(503).json({ success: false, error: 'FL server not reachable. Start it from the dashboard first.' });
    }
  });

  if (req.body && Object.keys(req.body).length) {
    const body = JSON.stringify(req.body);
    proxy.setHeader('Content-Type',   'application/json');
    proxy.setHeader('Content-Length', Buffer.byteLength(body));
    proxy.write(body);
  } else {
    req.pipe(proxy);
  }

  proxy.end();
});

// ── Start ─────────────────────────────────────────────────────────────────────

const server = app.listen(PORT, () => {
  console.log('');
  console.log('  BFL Healthcare Launcher');
  console.log(`  Open: http://localhost:${PORT}`);
  console.log('');
  console.log('  Use the dashboard to start IPFS, Fabric, and the FL server.');
  console.log('  No more terminals needed after this.');
  console.log('');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n  ERROR: Port ${PORT} is already in use.`);
    console.error(`  Run this to free it:  npx kill-port ${PORT}`);
    console.error(`  Or kill node:         taskkill /F /IM node.exe\n`);
  } else {
    console.error(`\n  Server error: ${err.message}\n`);
  }
  process.exit(1);
});
