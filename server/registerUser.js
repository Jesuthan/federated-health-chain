'use strict';

const FabricCAServices = require('fabric-ca-client');
const { Wallets } = require('fabric-network');
const fs = require('fs');
const path = require('path');

async function main() {
    try {
        const HOME = process.env.HOME || process.env.USERPROFILE;
        const wslBase = '\\\\wsl$\\Ubuntu\\home\\smart_touch_pc\\fabric-samples';
        const ccpBase = fs.existsSync(path.join(wslBase, 'test-network')) ? wslBase
            : fs.existsSync('C:\\fabric-samples\\test-network') ? 'C:\\fabric-samples'
            : path.join(HOME, 'fabric-samples');
        const ccpPath = path.join(
            ccpBase, 'test-network', 'organizations',
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

        // Always re-enroll appUser — the CA issues fresh crypto on every network restart,
        // so any wallet entry from a previous run is stale and causes "access denied".
        const userIdentity = await wallet.get('appUser');
        if (userIdentity) {
            console.log('appUser identity already in wallet — re-enrolling to match current CA.');
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

        // Persist the secret alongside the wallet so re-enrollment works after wallet clear
        const secretFile = path.join(walletPath, '.appuser-secret');

        let enrollSecret;
        try {
            enrollSecret = await ca.register(
                { affiliation: 'org1.department1', enrollmentID: 'appUser', role: 'client', maxEnrollments: -1 },
                adminUser
            );
            fs.writeFileSync(secretFile, enrollSecret, 'utf8');
        } catch (regErr) {
            if (!regErr.message.includes('74')) throw regErr;
            // Already registered — use the saved secret from the last registration
            if (fs.existsSync(secretFile)) {
                enrollSecret = fs.readFileSync(secretFile, 'utf8').trim();
                console.log('appUser already registered — using saved secret to re-enroll.');
            } else {
                throw new Error(
                    'appUser is registered in the CA but the secret file is missing.\n' +
                    'Restart the Fabric network (Stop → Start in dashboard) to reset the CA, then try again.'
                );
            }
        }

        // Enroll the user
        const enrollment = await ca.enroll({
            enrollmentID: 'appUser',
            enrollmentSecret: enrollSecret,
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
