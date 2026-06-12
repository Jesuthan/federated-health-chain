'use strict';

const { Contract } = require('fabric-contract-api');

// Composite key namespaces used for LevelDB-compatible range queries
const CK_BY_ROUND_MODEL = 'updByRM';    // [round, modelType, updateId]
const CK_BY_ROUND       = 'updByRound'; // [round, updateId]
const CK_BY_SENDER      = 'updBySender';// [sender, updateId]

class ModelRegistry extends Contract {

    async initLedger(ctx) {
        console.log('ModelRegistry chaincode initialized');
    }

    // ─── Client Updates ──────────────────────────────────────────────────────────

    /**
     * Store a federated learning model update (weight delta) on the ledger.
     * Also writes three composite-key index entries so we can range-query by
     * round+modelType, round, or sender without requiring CouchDB.
     */
    async storeUpdate(ctx, updateId, sender, modelType, round, ipfsCID, clipValue, noiseScale) {
        const existing = await ctx.stub.getState(updateId);
        if (existing && existing.length > 0) {
            throw new Error(`Update '${updateId}' already exists on the ledger`);
        }

        const txTs = ctx.stub.getTxTimestamp();
        const timestamp = new Date(Number(txTs.seconds) * 1000).toISOString();
        const roundInt = parseInt(round, 10);

        const record = {
            docType:    'modelUpdate',
            updateId,
            sender,
            modelType,
            round:      roundInt,
            ipfsCID,
            timestamp,
            clipValue:  parseFloat(clipValue),
            noiseScale: parseFloat(noiseScale),
        };

        const encoded = Buffer.from(JSON.stringify(record));

        // Primary record
        await ctx.stub.putState(updateId, encoded);

        // Composite-key indexes (value = full record for direct iteration)
        const roundStr = String(roundInt);
        await ctx.stub.putState(
            ctx.stub.createCompositeKey(CK_BY_ROUND_MODEL, [roundStr, modelType, updateId]),
            encoded,
        );
        await ctx.stub.putState(
            ctx.stub.createCompositeKey(CK_BY_ROUND, [roundStr, updateId]),
            encoded,
        );
        await ctx.stub.putState(
            ctx.stub.createCompositeKey(CK_BY_SENDER, [sender, updateId]),
            encoded,
        );

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

    /** Return all stored model updates (LevelDB-compatible range scan). */
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

    /** Query updates by FL round (LevelDB-compatible via composite key). */
    async queryByRound(ctx, round) {
        return this._queryComposite(ctx, CK_BY_ROUND, [String(parseInt(round, 10))]);
    }

    /** Query updates by sender (LevelDB-compatible via composite key). */
    async queryBySender(ctx, sender) {
        return this._queryComposite(ctx, CK_BY_SENDER, [sender]);
    }

    /**
     * Query updates for a specific round AND model type.
     * Used by server.js to count submissions and trigger aggregation.
     * LevelDB-compatible via composite key.
     */
    async queryByRoundAndModel(ctx, round, modelType) {
        return this._queryComposite(ctx, CK_BY_ROUND_MODEL, [String(parseInt(round, 10)), modelType]);
    }

    // ─── Global Model ─────────────────────────────────────────────────────────────

    /**
     * Store the aggregated global model CID produced by FedAvg.
     * Two keys are written:
     *   globalModel_{modelType}_round{round}  ← immutable per-round record
     *   globalModel_{modelType}_current       ← mutable pointer to latest round
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

    /** Return the latest global model record for a given model type. */
    async getLatestGlobalModel(ctx, modelType) {
        const key  = `globalModel_${modelType}_current`;
        const data = await ctx.stub.getState(key);
        if (!data || data.length === 0) {
            throw new Error(`No global model found for model type '${modelType}'`);
        }
        return data.toString();
    }

    /** Return the global model for a specific round + model type. */
    async getGlobalModelByRound(ctx, round, modelType) {
        const key  = `globalModel_${modelType}_round${parseInt(round, 10)}`;
        const data = await ctx.stub.getState(key);
        if (!data || data.length === 0) {
            throw new Error(`No global model for model='${modelType}' round='${round}'`);
        }
        return data.toString();
    }

    // ─── Internal ─────────────────────────────────────────────────────────────────

    async _queryComposite(ctx, namespace, partialAttrs) {
        const iterator = await ctx.stub.getStateByPartialCompositeKey(namespace, partialAttrs);
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
