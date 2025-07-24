import bittensor as bt
import os
import argparse

def main(config):
    # Configure your wallet and the subtensor
    try:
        print(f"config: {config}")
        wallet_name = getattr(config, 'wallet.name')
        wallet_hotkey = getattr(config, 'wallet.hotkey')

        wallet = bt.wallet(name=wallet_name, hotkey=wallet_hotkey)
        subtensor = bt.subtensor(network="finney")
        my_public_hash = "97dfad3ad4c74c3a94f508d81e57681d"

        hotkey_address = wallet.hotkey.ss58_address       
        r2_public_url = f"https://pub-{my_public_hash}.r2.dev/{hotkey_address}"

        # Commit the URL on-chain
        subtensor.commit(wallet=wallet, netuid=123, data=r2_public_url) # Use the correct netuid
        print("commit successful!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MANTIS commit tool"
    )
    parser.add_argument('--wallet.name', type=str, required=True, help='Bittensor wallet coldkey name that will be used to register')
    parser.add_argument('--wallet.hotkey', type=str, required=True, help='Bittensor wallet hotkey name that will be used to register')
    config = parser.parse_args()
    main(config)