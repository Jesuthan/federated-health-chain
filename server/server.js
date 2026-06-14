'use strict';

const express    = require('express');
const { Wallets, Gateway } = require('fabric-network');
const { v4: uuidv4 }       = require('uuid');
const { spawn }            = require('child_process');
const fs   = require('fs');
const path = require('path');

// ─── Metrics store (persisted to metrics.json) ─────────────────────────────────
const METRICS_FILE = path.join(__dirname, 'metrics.json');
let _metrics = [];
try {
    if (fs.existsSync(METRICS_FILE)) {
        _metrics = JSON.parse(fs.readFileSync(METRICS_FILE, 'utf8'));
    }
} catch (_) { _metrics = []; }

function saveMetrics() {
    try { fs.writeFileSync(METRICS_FILE, JSON.stringify(_metrics, null, 2)); } catch (_) {}
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

const PORT           = process.env.PORT        || 3000;
const CHANNEL_NAME   = 'mychannel';
const CHAINCODE_NAME = 'modelregistry';

// How many hospital updates per round+model trigger FedAvg aggregation.
const MIN_CLIENTS = parseInt(process.env.MIN_CLIENTS || '3', 10);

const HOME    = process.env.HOME || process.env.USERPROFILE;

// Fabric network runs via WSL2 — certs are in the WSL2 Linux filesystem.
// \\wsl$\Ubuntu\home\<user>\fabric-samples has the fresh TLS certs from this run.
// Fall back to C:\fabric-samples (Windows) only if WSL2 path doesn't exist.
function findCcpBase() {
    const wslBase = '\\\\wsl$\\Ubuntu\\home\\smart_touch_pc\\fabric-samples';
    if (fs.existsSync(path.join(wslBase, 'test-network'))) return wslBase;
    if (fs.existsSync('C:\\fabric-samples\\test-network')) return 'C:\\fabric-samples';
    return path.join(HOME, 'fabric-samples');
}
const ccpBase = findCcpBase();
const ccpPath = path.join(
    ccpBase, 'test-network', 'organizations',
    'peerOrganizations', 'org1.example.com',
    'connection-org1.json'
);

// ─── Fabric helper ─────────────────────────────────────────────────────────────

let _fabricWarned = false;
let _walletWarned = false;
const _routeWarned = {};   // key: route, value: last error message shown

function warnOnce(route, msg) {
    if (_routeWarned[route] === msg) return;
    _routeWarned[route] = msg;
    console.error(`${route} error:`, msg);
}
function clearWarn(route) { delete _routeWarned[route]; }
async function getContract() {
    if (!fs.existsSync(ccpPath)) {
        if (!_fabricWarned) {
            console.warn(`[Fabric] Connection profile not found: ${ccpPath}`);
            console.warn('[Fabric] Start the Fabric network first.');
            _fabricWarned = true;
        }
        throw new Error('Fabric network not running. Start it from the dashboard.');
    }
    _fabricWarned = false;
    const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    // Inject orderer + Org2 peer so the SDK can route transactions without discovery.
    // The generated connection-org1.json omits these; we add them from known test-network ports.
    const ordererTLS = fs.readFileSync(
        path.join(ccpBase, 'test-network', 'organizations',
            'ordererOrganizations', 'example.com', 'orderers',
            'orderer.example.com', 'msp', 'tlscacerts', 'tlsca.example.com-cert.pem'),
        'utf8'
    );
    const org2TLS = fs.readFileSync(
        path.join(ccpBase, 'test-network', 'organizations',
            'peerOrganizations', 'org2.example.com', 'tlsca',
            'tlsca.org2.example.com-cert.pem'),
        'utf8'
    );

    ccp.orderers = {
        'orderer.example.com': {
            url: 'grpcs://localhost:7050',
            tlsCACerts: { pem: ordererTLS },
            grpcOptions: {
                'ssl-target-name-override': 'orderer.example.com',
                'hostnameOverride':         'orderer.example.com',
            },
        },
    };
    ccp.peers['peer0.org2.example.com'] = {
        url: 'grpcs://localhost:9051',
        tlsCACerts: { pem: org2TLS },
        grpcOptions: {
            'ssl-target-name-override': 'peer0.org2.example.com',
            'hostnameOverride':         'peer0.org2.example.com',
        },
    };
    ccp.organizations['Org2'] = {
        mspid: 'Org2MSP',
        peers: ['peer0.org2.example.com'],
    };
    ccp.channels = {
        mychannel: {
            orderers: ['orderer.example.com'],
            peers: {
                'peer0.org1.example.com': { endorsingPeer: true, chaincodeQuery: true, ledgerQuery: true, eventSource: true },
                'peer0.org2.example.com': { endorsingPeer: true, chaincodeQuery: true, ledgerQuery: true, eventSource: false },
            },
        },
    };

    const walletPath = path.join(__dirname, 'wallet');
    const wallet     = await Wallets.newFileSystemWallet(walletPath);

    const identity = await wallet.get('appUser');
    if (!identity) {
        if (!_walletWarned) {
            console.warn('[Fabric] appUser not in wallet — click "Setup Wallet" in the dashboard.');
            _walletWarned = true;
        }
        throw new Error('appUser identity not found in wallet. Run registerUser.js first.');
    }
    _walletWarned = false;

    const gateway = new Gateway();
    await gateway.connect(ccp, {
        wallet,
        identity:  'appUser',
        discovery: { enabled: false },
    });

    const network  = await gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);
    return { gateway, contract };
}

// ─── Aggregation trigger ───────────────────────────────────────────────────────

/**
 * Spawn the Python FedAvg aggregator as a detached background process.
 * Non-blocking — the POST /api/updates response is already sent before this.
 */
function triggerAggregation(round, modelType, algo = 'fedprox', mu = 0.01) {
    const aggregatorPath = path.join(__dirname, 'aggregator.py');
    const serverUrl      = `http://localhost:${PORT}`;

    console.log(`\n[AGG] Threshold reached for model=${modelType} round=${round} algo=${algo}`);
    console.log(`[AGG] Spawning aggregator: ${aggregatorPath}`);

    const proc = spawn('python', [
        aggregatorPath,
        '--round',  String(round),
        '--model',  modelType,
        '--algo',   algo,
        '--mu',     String(mu),
        '--server', serverUrl,
    ], {
        detached: true,
        stdio:    'inherit',
        env:      { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    });

    proc.on('error', (err) => {
        console.error(`[AGG] Failed to start aggregator: ${err.message}`);
    });

    proc.unref();  // let Node.js exit independently of the child
}

// ─── Routes — Client Updates ───────────────────────────────────────────────────

/** GET /health */
app.get('/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

/** GET /api/metrics — all round metrics for convergence charts */
app.get('/api/metrics', (_req, res) => {
    res.json({ success: true, count: _metrics.length, metrics: _metrics });
});

/** POST /api/metrics — called by aggregator.py after each round */
app.post('/api/metrics', (req, res) => {
    const { round, modelType, accuracy, algorithm, mu, clientCount, epsilon, timestamp } = req.body;
    if (round == null || !modelType || accuracy == null) {
        return res.status(400).json({ success: false, error: 'Missing required fields' });
    }
    const entry = {
        round: parseInt(round, 10), modelType, accuracy: parseFloat(accuracy),
        algorithm: algorithm || 'fedavg', mu: parseFloat(mu || 0),
        clientCount: parseInt(clientCount || 0, 10),
        epsilon: parseFloat(epsilon || 0),
        timestamp: timestamp || new Date().toISOString(),
    };
    // Replace existing entry for same round+model+algorithm, or append
    const idx = _metrics.findIndex(
        m => m.round === entry.round && m.modelType === entry.modelType && m.algorithm === entry.algorithm
    );
    if (idx >= 0) _metrics[idx] = entry; else _metrics.push(entry);
    saveMetrics();
    console.log(`[Metrics] ${entry.modelType} round ${entry.round} ${entry.algorithm.toUpperCase()} accuracy=${(entry.accuracy*100).toFixed(2)}% ε=${entry.epsilon}`);
    res.status(201).json({ success: true, entry });
});

/** GET /api/updates — all model updates */
app.get('/api/updates', async (_req, res) => {
    let gateway;
    try {
        const { gateway: gw, contract } = await getContract();
        gateway = gw;
        const data    = await contract.evaluateTransaction('getAllUpdates');
        const updates = JSON.parse(data.toString());
        res.json({ success: true, count: updates.length, updates });
    } catch (err) {
        warnOnce('GET /api/updates', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/** GET /api/updates/round/:round */
app.get('/api/updates/round/:round', async (req, res) => {
    let gateway;
    try {
        const { gateway: gw, contract } = await getContract();
        gateway = gw;
        const data    = await contract.evaluateTransaction('queryByRound', req.params.round);
        const updates = JSON.parse(data.toString());
        res.json({ success: true, round: parseInt(req.params.round, 10), count: updates.length, updates });
    } catch (err) {
        warnOnce('GET /api/updates/round', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/** GET /api/updates/sender/:sender */
app.get('/api/updates/sender/:sender', async (req, res) => {
    let gateway;
    try {
        const { gateway: gw, contract } = await getContract();
        gateway = gw;
        const data    = await contract.evaluateTransaction('queryBySender', req.params.sender);
        const updates = JSON.parse(data.toString());
        res.json({ success: true, sender: req.params.sender, count: updates.length, updates });
    } catch (err) {
        warnOnce('GET /api/updates/sender', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/**
 * POST /api/updates
 * Body: { sender, modelType, round, ipfsCID, clipValue?, noiseScale? }
 *
 * After storing, counts how many updates exist for this round+modelType.
 * If the count reaches MIN_CLIENTS, the FedAvg aggregator is auto-triggered.
 */
app.post('/api/updates', async (req, res) => {
    let gateway;
    try {
        const { sender, modelType, round, ipfsCID,
                clipValue = 1.0, noiseScale = 0.1,
                sampleCount = 1000, algorithm = 'fedprox', mu = 0.01 } = req.body;

        if (!sender || !modelType || round == null || !ipfsCID) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: sender, modelType, round, ipfsCID',
            });
        }

        const { gateway: gw, contract } = await getContract();
        gateway = gw;

        // Reject duplicate submission from the same hospital for the same round+model
        const existingRaw = await contract.evaluateTransaction('queryByRoundAndModel', String(round), modelType);
        const existing = JSON.parse(existingRaw.toString());
        const duplicate = existing.find(u => u.sender === sender);
        if (duplicate) {
            return res.status(409).json({
                success: false,
                error: `${sender} already submitted for ${modelType} round ${round}. Each hospital can only submit once per round.`,
            });
        }

        const id = `update_${sender}_round${round}_${uuidv4().slice(0, 8)}`;

        await contract.submitTransaction(
            'storeUpdate',
            id,
            sender,
            modelType,
            String(round),
            ipfsCID,
            String(clipValue),
            String(noiseScale),
        );

        console.log(`Stored: ${id} | sender: ${sender} | model: ${modelType} | round: ${round}`);

        // Count how many updates now exist for this round + model type.
        const countData = await contract.evaluateTransaction(
            'queryByRoundAndModel', String(round), modelType
        );
        const updates     = JSON.parse(countData.toString());
        const updateCount = updates.length;

        console.log(`Round ${round} ${modelType}: ${updateCount}/${MIN_CLIENTS} update(s) received`);

        res.status(201).json({
            message:      'Model update stored on blockchain',
            id,
            sender,
            modelType,
            round,
            ipfsCID,
            updateCount,
            minClients:   MIN_CLIENTS,
            aggregating:  updateCount >= MIN_CLIENTS,
        });

        // Trigger aggregation AFTER responding so the client is not blocked.
        if (updateCount >= MIN_CLIENTS) {
            // Guard: only trigger once — check no global model already exists for this round.
            try {
                await contract.evaluateTransaction('getGlobalModelByRound', String(round), modelType);
                console.log(`[AGG] Global model for round ${round} ${modelType} already exists — skipping.`);
            } catch (_) {
                // Expected: no global model yet → safe to aggregate.
                triggerAggregation(round, modelType, algorithm, parseFloat(mu));
            }
        }

    } catch (err) {
        warnOnce('POST /api/updates', err.message);
        if (!res.headersSent) {
            res.status(500).json({ success: false, error: err.message });
        }
    } finally {
        if (gateway) gateway.disconnect();
    }
});

// ─── Routes — Global Model ─────────────────────────────────────────────────────

/**
 * POST /api/global-model
 * Body: { round, modelType, ipfsCID, clientCount }
 * Called by aggregator.py after FedAvg completes.
 */
app.post('/api/global-model', async (req, res) => {
    let gateway;
    try {
        const { round, modelType, ipfsCID, clientCount = 0 } = req.body;

        if (!round == null || !modelType || !ipfsCID) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: round, modelType, ipfsCID',
            });
        }

        const { gateway: gw, contract } = await getContract();
        gateway = gw;

        const id = `globalModel_${modelType}_round${round}`;

        await contract.submitTransaction(
            'storeGlobalModel',
            String(round),
            modelType,
            ipfsCID,
            String(clientCount),
        );

        console.log(`Global model stored: round=${round} model=${modelType} CID=${ipfsCID}`);
        res.status(201).json({
            message: 'Global model recorded on blockchain',
            id,
            round,
            modelType,
            ipfsCID,
            clientCount,
        });
    } catch (err) {
        warnOnce('POST /api/global-model', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/**
 * GET /api/global-model/:modelType/latest
 * Returns the CID of the most recent global model for a model type.
 * Used by hospitals (fl_client.py --pull-global) to get the latest weights.
 */
app.get('/api/global-model/:modelType/latest', async (req, res) => {
    let gateway;
    try {
        const { modelType } = req.params;
        const { gateway: gw, contract } = await getContract();
        gateway = gw;

        const data   = await contract.evaluateTransaction('getLatestGlobalModel', modelType);
        const record = JSON.parse(data.toString());
        res.json({ success: true, ...record });
    } catch (err) {
        const notFound = err.message.includes('No global model found');
        if (!notFound) warnOnce('GET /api/global-model/latest', err.message);
        res.status(notFound ? 404 : 500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/**
 * GET /api/global-model/:modelType/round/:round
 * Returns the global model record for a specific round.
 */
app.get('/api/global-model/:modelType/round/:round', async (req, res) => {
    let gateway;
    try {
        const { modelType, round } = req.params;
        const { gateway: gw, contract } = await getContract();
        gateway = gw;

        const data   = await contract.evaluateTransaction('getGlobalModelByRound', round, modelType);
        const record = JSON.parse(data.toString());
        res.json({ success: true, ...record });
    } catch (err) {
        const notFound = err.message.includes('No global model');
        warnOnce('GET /api/global-model/round', err.message);
        res.status(notFound ? 404 : 500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

// ─── Routes — Simulator (browser-triggered hospital client) ───────────────────

/**
 * POST /api/simulate
 * Body: { sender, modelType, round, clip?, noise?, pullGlobal? }
 * Spawns fl_client.py and returns stdout when done.
 */
app.post('/api/simulate', (req, res) => {
    const { sender, modelType, round, clip = 1.0, noise = 0.1,
            pullGlobal = false, algo = 'fedprox', mu = 0.01, samples = 64 } = req.body;

    if (!sender || !modelType || round == null) {
        return res.status(400).json({ success: false, error: 'Missing required fields: sender, modelType, round' });
    }

    const clientScript = path.join(__dirname, '..', 'client', 'fl_client.py');
    const args = [
        clientScript,
        '--sender',  sender,
        '--model',   modelType,
        '--round',   String(round),
        '--clip',    String(clip),
        '--noise',   String(noise),
        '--algo',    algo,
        '--mu',      String(mu),
        '--samples', String(samples),
        '--server',  `http://localhost:${PORT}`,
    ];
    if (pullGlobal) args.push('--pull-global');

    console.log(`[SIM] Running: python ${args.join(' ')}`);

    const proc = spawn('python', args, {
        env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    });
    let output = '';
    let errors = '';

    proc.stdout.on('data', d => { output += d.toString('utf8'); });
    proc.stderr.on('data', d => { errors += d.toString('utf8'); });

    proc.on('close', code => {
        const success = code === 0;
        console.log(`[SIM] ${sender} ${modelType} round ${round} exited with code ${code}`);
        res.json({ success, output, error: success ? null : errors });
    });

    proc.on('error', err => {
        res.status(500).json({ success: false, error: `Failed to spawn client: ${err.message}` });
    });
});

/**
 * POST /api/aggregate-now
 * Body: { round, modelType }
 * Manually trigger FedAvg aggregation (bypasses MIN_CLIENTS check).
 */
app.post('/api/aggregate-now', (req, res) => {
    const { round, modelType } = req.body;

    if (round == null || !modelType) {
        return res.status(400).json({ success: false, error: 'Missing required fields: round, modelType' });
    }

    triggerAggregation(round, modelType);
    res.json({ success: true, message: `FedAvg aggregation triggered for ${modelType} round ${round}` });
});

// ─── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
    console.log(`\nFederated Learning REST Server`);
    console.log(`  Listening on  : http://localhost:${PORT}`);
    console.log(`  MIN_CLIENTS   : ${MIN_CLIENTS}  (set MIN_CLIENTS env var to change)`);
    console.log(`\nClient update endpoints:`);
    console.log(`  GET  /health`);
    console.log(`  GET  /api/updates`);
    console.log(`  POST /api/updates`);
    console.log(`  GET  /api/updates/round/:round`);
    console.log(`  GET  /api/updates/sender/:sender`);
    console.log(`\nGlobal model endpoints:`);
    console.log(`  POST /api/global-model`);
    console.log(`  GET  /api/global-model/:modelType/latest`);
    console.log(`  GET  /api/global-model/:modelType/round/:round\n`);
});
