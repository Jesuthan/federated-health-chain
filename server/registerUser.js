'use strict';

const FabricCAServices = require('fabric-ca-client');
const { Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

async function main() {
    try {
        const ccpPath = path.resolve(
            process.env.HOME,
            'fedlearn-fabric', 'fabric-samples', 'test-network', 'organizations',
            'peerOrganizations', 'org1.example.com',
            'connection-org1.json'
        );

        if (!fs.existsSync(ccpPath)) {
            throw new Error(`Connection profile not found at: ${ccpPath}\nMake sure the test-network is running.`);
        }

        const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

        // Create CA client for Org1
        const caInfo = ccp.certificateAuthorities['ca.org1.example.com'];
        const caTLSCACerts = caInfo.tlsCACerts.pem;
        const ca = new FabricCAServices(
            caInfo.url,
            { trustedRoots: caTLSCACerts, verify: false },
            caInfo.caName
        );

        const walletPath = path.join(process.cwd(), 'wallet');
        const wallet = await Wallets.newFileSystemWallet(walletPath);
        console.log(`Wallet path: ${walletPath}`);

        // Check if appUser already registered
        const userIdentity = await wallet.get('appUser');
        if (userIdentity) {
            console.log('appUser identity already exists in wallet. Skipping registration.');
            return;
        }

        // Admin must exist first
        const adminIdentity = await wallet.get('admin');
        if (!adminIdentity) {
            console.error('Admin identity not found in wallet. Run enrollAdmin.js first.');
            process.exit(1);
        }

        // Build admin user context
        const provider = wallet.getProviderRegistry().getProvider(adminIdentity.type);
        const adminUser = await provider.getUserContext(adminIdentity, 'admin');

        // Register the user
        const secret = await ca.register(
            {
                affiliation: 'org1.department1',
                enrollmentID: 'appUser',
                role: 'client',
            },
            adminUser
        );

        // Enroll the user
        const enrollment = await ca.enroll({
            enrollmentID: 'appUser',
            enrollmentSecret: secret,
        });

        const x509Identity = {
            credentials: {
                certificate: enrollment.certificate,
                privateKey: enrollment.key.toBytes(),
            },
            mspId: 'Org1MSP',
            type: 'X.509',
        };

        await wallet.put('appUser', x509Identity);
        console.log('Successfully registered and enrolled appUser; identity stored in wallet.');

    } catch (error) {
        console.error(`Failed to register user: ${error}`);
        process.exit(1);
    }
}

main();
