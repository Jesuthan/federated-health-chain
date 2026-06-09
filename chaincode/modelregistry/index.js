'use strict';

const { Contract } = require('fabric-contract-api');

class ModelRegistry extends Contract {

    async initLedger(ctx) {
        console.log('ModelRegistry chaincode initialized');
    }

    // ─── Client Updates ──────────────────────────────────────────────────────────

    /**
     * Store a federated learning model update (weight delta) on the ledger.
     * @param {string} updateId   - Unique ID for this update
     * @param {string} sender     - Client identifier (e.g. "Hospital1")
     * @param {string} modelType  - "covid" | "skin"
     * @param {string} round      - FL round number
     * @param {string} ipfsCID    - IPFS CID of the delta .pt file
     * @param {string} clipValue  - DP gradient clip norm
     * @param {string} noiseScale - DP Gaussian noise scale
     */
    async storeUpdate(ctx, updateId, sender, modelType, round, ipfsCID, clipValue, noiseScale) {
        const existing = await ctx.stub.getState(updateId);
        if (existing && existing.length > 0) {
            throw new Error(`Update '${updateId}' already exists on the ledger`);
        }

        const txTs = ctx.stub.getTxTimestamp();
        const timestamp = new Date(Number(txTs.seconds) * 1000).toISOString();

        const record = {
            docType:    'modelUpdate',
            updateId,
            sender,
            modelType,
            round:      parseInt(round, 10),
            ipfsCID,
            timestamp,
            clipValue:  parseFloat(clipValue),
            noiseScale: parseFloat(noiseScale),
        };

        await ctx.stub.putState(updateId, Buffer.from(JSON.stringify(record)));
        console.log(`Stored update: ${updateId} | sender: ${sender} | model: ${modelType} | round: ${round}`);
        return JSON.stringify(record);
    }

    /** Retrieve a single update by ID. */
    async getUpdate(ctx, updateId) {
        const data = await ctx.stub.getState(updateId);
        if (!data || data.length === 0) {
            throw new Error(`Update '${updateId}' does not exist`);
        }
        return data.toString();
    }

    /** Return all stored model updates. */
    async getAllUpdates(ctx) {
        const iterator = await ctx.stub.getStateByRange('', '');
        const results  = [];
        let result     = await iterator.next();

        while (!result.done) {
            const str = result.value.value.toString('utf8');
            try {
                const rec = JSON.parse(str);
                if (rec.docType === 'modelUpdate') results.push(rec);
            } catch (_) {}
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }

    /** Query updates by FL round (requires CouchDB). */
    async queryByRound(ctx, round) {
        const query = {
            selector: { docType: 'modelUpdate', round: parseInt(round, 10) },
        };
        return this._runQuery(ctx, query);
    }

    /** Query updates by sender (requires CouchDB). */
    async queryBySender(ctx, sender) {
        const query = {
            selector: { docType: 'modelUpdate', sender },
        };
        return this._runQuery(ctx, query);
    }

    /**
     * Query updates for a specific round AND model type.
     * Used by server.js to count submissions and trigger aggregation.
     * Requires CouchDB.
     */
    async queryByRoundAndModel(ctx, round, modelType) {
        const query = {
            selector: {
                docType:   'modelUpdate',
                round:     parseInt(round, 10),
                modelType,
            },
        };
        return this._runQuery(ctx, query);
    }

    // ─── Global Model ─────────────────────────────────────────────────────────────

    /**
     * Store the aggregated global model CID produced by FedAvg.
     * Two keys are written:
     *   globalModel_{modelType}_round{round}  ← immutable per-round record
     *   globalModel_{modelType}_current       ← mutable pointer to latest round
     *
     * @param {string} round        - FL round number
     * @param {string} modelType    - "covid" | "skin"
     * @param {string} ipfsCID      - IPFS CID of the global model .pt file
     * @param {string} clientCount  - number of clients merged in this round
     */
    async storeGlobalModel(ctx, round, modelType, ipfsCID, clientCount) {
        const roundInt = parseInt(round, 10);

        const txTs = ctx.stub.getTxTimestamp();
        const timestamp = new Date(Number(txTs.seconds) * 1000).toISOString();

        const record = {
            docType:     'globalModel',
            round:       roundInt,
            modelType,
            ipfsCID,
            clientCount: parseInt(clientCount, 10),
            timestamp,
        };

        const perRoundKey = `globalModel_${modelType}_round${roundInt}`;
        const currentKey  = `globalModel_${modelType}_current`;

        await ctx.stub.putState(perRoundKey, Buffer.from(JSON.stringify(record)));
        await ctx.stub.putState(currentKey,  Buffer.from(JSON.stringify(record)));

        console.log(`Stored global model: round=${round} model=${modelType} CID=${ipfsCID}`);
        return JSON.stringify(record);
    }

    /**
     * Return the latest global model record for a given model type.
     * Reads the mutable `globalModel_{modelType}_current` key.
     */
    async getLatestGlobalModel(ctx, modelType) {
        const key  = `globalModel_${modelType}_current`;
        const data = await ctx.stub.getState(key);
        if (!data || data.length === 0) {
            throw new Error(`No global model found for model type '${modelType}'`);
        }
        return data.toString();
    }

    /**
     * Return the global model for a specific round + model type.
     */
    async getGlobalModelByRound(ctx, round, modelType) {
        const key  = `globalModel_${modelType}_round${parseInt(round, 10)}`;
        const data = await ctx.stub.getState(key);
        if (!data || data.length === 0) {
            throw new Error(`No global model for model='${modelType}' round='${round}'`);
        }
        return data.toString();
    }

    // ─── Internal ─────────────────────────────────────────────────────────────────

    async _runQuery(ctx, queryObject) {
        const iterator = await ctx.stub.getQueryResult(JSON.stringify(queryObject));
        const results  = [];
        let result     = await iterator.next();

        while (!result.done) {
            try {
                results.push(JSON.parse(result.value.value.toString('utf8')));
            } catch (_) {}
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }
}

module.exports.contracts = [ModelRegistry];
