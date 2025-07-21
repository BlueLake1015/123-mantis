import bittensor as bt
import os
from dotenv import load_dotenv

load_dotenv() # Load credentials from .env file

# Configure your wallet and the subtensor
wallet = bt.wallet(name=os.getenv("WALLET_NAME"), hotkey=os.getenv("HOTKEY_NAME"))
subtensor = bt.subtensor(network="finney")

# The public URL of your object in R2
# NOTE: The public URL format may vary slightly based on your R2 setup.
# Ensure your bucket is public and the URL is correct.
my_public_hash = "02425efe69384292aee6a505da4df9eb"
hotkey = os.getenv("HOTKEY")
r2_public_url = f"https://pub-{my_public_hash}.r2.dev/{hotkey}"

# Commit the URL on-chain
subtensor.commit(wallet=wallet, netuid=123, data=r2_public_url) # Use the correct netuid
print("commit successful!")