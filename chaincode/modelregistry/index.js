'use strict';

const { Contract } = require('fabric-contract-api');

class ModelRegistry extends Contract {

    async initLedger(ctx) {
        console.log('ModelRegistry chaincode initialized');
    }

    /**
     * Store a federated learning model update on the ledger.
     * @param {Context} ctx
     * @param {string} updateId   - Unique ID for this update
     * @param {string} sender     - Client identifier (e.g. "Client1")
     * @param {string} modelType  - "covid" | "skin"
     * @param {string} round      - FL round number (string, parsed to int)
     * @param {string} ipfsCID    - IPFS Content Identifier of the model update
     * @param {string} timestamp  - ISO timestamp
     * @param {string} clipValue  - DP gradient clip norm
     * @param {string} noiseScale - DP Gaussian noise scale
     */
    async storeUpdate(ctx, updateId, sender, modelType, round, ipfsCID, clipValue, noiseScale) {
        // Reject duplicate IDs
        const existing = await ctx.stub.getState(updateId);
        if (existing && existing.length > 0) {
            throw new Error(`Update with ID '${updateId}' already exists on the ledger`);
        }

        const record = {
            docType: 'modelUpdate',
            updateId,
            sender,
            modelType,
            round: parseInt(round, 10),
            ipfsCID,
            timestamp: new Date().toISOString(),
            clipValue: parseFloat(clipValue),
            noiseScale: parseFloat(noiseScale),
        };

        await ctx.stub.putState(updateId, Buffer.from(JSON.stringify(record)));
        console.log(`Stored update: ${updateId} | sender: ${sender} | round: ${round}`);
        return JSON.stringify(record);
    }

    /** Retrieve a single update by its ID. */
    async getUpdate(ctx, updateId) {
        const data = await ctx.stub.getState(updateId);
        if (!data || data.length === 0) {
            throw new Error(`Update ${updateId} does not exist`);
        }
        return data.toString();
    }

    /** Return all stored model updates. */
    async getAllUpdates(ctx) {
        const iterator = await ctx.stub.getStateByRange('', '');
        const results = [];

        let result = await iterator.next();
        while (!result.done) {
            const strValue = result.value.value.toString('utf8');
            try {
                const record = JSON.parse(strValue);
                if (record.docType === 'modelUpdate') {
                    results.push(record);
                }
            } catch (_) {
                // skip malformed entries
            }
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }

    /** Query updates by FL round number. Requires CouchDB state database. */
    async queryByRound(ctx, round) {
        const query = {
            selector: {
                docType: 'modelUpdate',
                round: parseInt(round, 10),
            },
        };

        const iterator = await ctx.stub.getQueryResult(JSON.stringify(query));
        const results = [];

        let result = await iterator.next();
        while (!result.done) {
            const strValue = result.value.value.toString('utf8');
            try {
                results.push(JSON.parse(strValue));
            } catch (_) {}
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }

    /** Query updates by sender (client ID). Requires CouchDB. */
    async queryBySender(ctx, sender) {
        const query = {
            selector: {
                docType: 'modelUpdate',
                sender,
            },
        };

        const iterator = await ctx.stub.getQueryResult(JSON.stringify(query));
        const results = [];

        let result = await iterator.next();
        while (!result.done) {
            const strValue = result.value.value.toString('utf8');
            try {
                results.push(JSON.parse(strValue));
            } catch (_) {}
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }
}

module.exports.contracts = [ModelRegistry];
