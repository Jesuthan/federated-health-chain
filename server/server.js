'use strict';

const express = require('express');
const { Wallets, Gateway } = require('fabric-network');
const { v4: uuidv4 } = require('uuid');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const PORT = process.env.PORT || 3000;
const CHANNEL_NAME = 'mychannel';
const CHAINCODE_NAME = 'modelregistry';

const ccpPath = path.resolve(
    process.env.HOME,
    'fedlearn-fabric', 'fabric-samples', 'test-network', 'organizations',
    'peerOrganizations', 'org1.example.com',
    'connection-org1.json'
);

// ─── Fabric helper ─────────────────────────────────────────────────────────────

async function getContract() {
    if (!fs.existsSync(ccpPath)) {
        throw new Error(`Connection profile not found: ${ccpPath}`);
    }
    const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

    const walletPath = path.join(__dirname, 'wallet');
    const wallet = await Wallets.newFileSystemWallet(walletPath);

    const identity = await wallet.get('appUser');
    if (!identity) {
        throw new Error('appUser identity not found in wallet. Run registerUser.js first.');
    }

    const gateway = new Gateway();
    await gateway.connect(ccp, {
        wallet,
        identity: 'appUser',
        discovery: { enabled: true, asLocalhost: true },
    });

    const network = await gateway.getNetwork(CHANNEL_NAME);
    const contract = network.getContract(CHAINCODE_NAME);

    return { gateway, contract };
}

// ─── Routes ────────────────────────────────────────────────────────────────────

/** GET /health */
app.get('/health', (_req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

/** GET /api/updates — all model updates */
app.get('/api/updates', async (_req, res) => {
    let gateway;
    try {
        const { gateway: gw, contract } = await getContract();
        gateway = gw;
        const data = await contract.evaluateTransaction('getAllUpdates');
        const updates = JSON.parse(data.toString());
        res.json({ success: true, count: updates.length, updates });
    } catch (err) {
        console.error('GET /api/updates error:', err.message);
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
        const data = await contract.evaluateTransaction('queryByRound', req.params.round);
        const updates = JSON.parse(data.toString());
        res.json({ success: true, round: parseInt(req.params.round, 10), count: updates.length, updates });
    } catch (err) {
        console.error('GET /api/updates/round error:', err.message);
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
        const data = await contract.evaluateTransaction('queryBySender', req.params.sender);
        const updates = JSON.parse(data.toString());
        res.json({ success: true, sender: req.params.sender, count: updates.length, updates });
    } catch (err) {
        console.error('GET /api/updates/sender error:', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

/**
 * POST /api/updates
 * Body: { sender, modelType, round, ipfsCID, clipValue?, noiseScale? }
 */
app.post('/api/updates', async (req, res) => {
    let gateway;
    try {
        const { sender, modelType, round, ipfsCID, clipValue = 1.0, noiseScale = 0.1 } = req.body;

        if (!sender || !modelType || round == null || !ipfsCID) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: sender, modelType, round, ipfsCID',
            });
        }

        const { gateway: gw, contract } = await getContract();
        gateway = gw;

        const id = `update_${sender}_round${round}_${uuidv4().slice(0, 8)}`;

        await contract.submitTransaction(
            'storeUpdate',
            id,
            sender,
            modelType,
            String(round),
            ipfsCID,
            String(clipValue),
            String(noiseScale)
        );

        console.log(`Stored: ${id} from ${sender} round ${round}`);
        res.status(201).json({ message: 'Model update stored on blockchain', id, sender, modelType, round, ipfsCID });
    } catch (err) {
        console.error('POST /api/updates error:', err.message);
        res.status(500).json({ success: false, error: err.message });
    } finally {
        if (gateway) gateway.disconnect();
    }
});

// ─── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
    console.log(`\nFederated Learning REST Server`);
    console.log(`  Listening on : http://localhost:${PORT}`);
    console.log(`\nEndpoints:`);
    console.log(`  GET  /health`);
    console.log(`  GET  /api/updates`);
    console.log(`  POST /api/updates`);
    console.log(`  GET  /api/updates/round/:round`);
    console.log(`  GET  /api/updates/sender/:sender\n`);
});
