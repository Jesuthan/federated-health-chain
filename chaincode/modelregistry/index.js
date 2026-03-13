'use strict';

const { Contract } = require('fabric-contract-api');

class ModelRegistry extends Contract {

    async initLedger(ctx) {
        console.log('ModelRegistry chaincode initialized');
    }

    async storeUpdate(ctx, updateId, sender, modelType, round, ipfsCID, clipValue, noiseScale) {
        const existing = await ctx.stub.getState(updateId);
        if (existing && existing.length > 0) {
            throw new Error(`Update with ID '${updateId}' already exists on the ledger`);
        }

        const ts = ctx.stub.getTxTimestamp();
        const timestamp = new Date(ts.seconds.low * 1000).toISOString();

        const record = {
            docType: 'modelUpdate',
            updateId,
            sender,
            modelType,
            round: parseInt(round, 10),
            ipfsCID,
            timestamp,
            clipValue: parseFloat(clipValue),
            noiseScale: parseFloat(noiseScale),
        };

        await ctx.stub.putState(updateId, Buffer.from(JSON.stringify(record)));
        console.log(`Stored update: ${updateId} | sender: ${sender} | round: ${round}`);
        return JSON.stringify(record);
    }

    async getUpdate(ctx, updateId) {
        const data = await ctx.stub.getState(updateId);
        if (!data || data.length === 0) {
            throw new Error(`Update ${updateId} does not exist`);
        }
        return data.toString();
    }

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
            } catch (_) {}
            result = await iterator.next();
        }

        await iterator.close();
        return JSON.stringify(results);
    }

    async queryByRound(ctx, round) {
        const query = { selector: { docType: 'modelUpdate', round: parseInt(round, 10) } };
        const iterator = await ctx.stub.getQueryResult(JSON.stringify(query));
        const results = [];
        let result = await iterator.next();
        while (!result.done) {
            try { results.push(JSON.parse(result.value.value.toString('utf8'))); } catch (_) {}
            result = await iterator.next();
        }
        await iterator.close();
        return JSON.stringify(results);
    }

    async queryBySender(ctx, sender) {
        const query = { selector: { docType: 'modelUpdate', sender } };
        const iterator = await ctx.stub.getQueryResult(JSON.stringify(query));
        const results = [];
        let result = await iterator.next();
        while (!result.done) {
            try { results.push(JSON.parse(result.value.value.toString('utf8'))); } catch (_) {}
            result = await iterator.next();
        }
        await iterator.close();
        return JSON.stringify(results);
    }
}

module.exports.contracts = [ModelRegistry];
