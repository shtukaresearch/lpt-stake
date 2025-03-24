# Script for fetching Livepeer staking data from Arbitrum.
# First, fetch daily Arbitrum block numbers at the same time

from datetime import datetime, timedelta
import os, sys
import urllib3
import requests
import time
import json
from pytz import UTC
from web3 import Web3

retries = urllib3.util.Retry(
    total=5,  # Total number of retries
    backoff_factor=1,  # Exponential backoff factor (e.g., 1, 2, 4, 8, 16 seconds)
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
)

def get_block_number_by_time(apikey: str, timestamp: datetime) -> int:
    url = "https://api.arbiscan.io/api"
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": int(timestamp.timestamp()),
        "closest": "before",
        "apikey": apikey
    }
    response = requests.get(url, params=params)
    data = response.json()
    if data["status"] == "1":
        return int(data["result"])
    else:
        raise ValueError(f"Error fetching block number: {data['message']}")



def fetch_arbitrum_daily_blocks(apikey: str, start: datetime, num_days: int) -> list[int]:
    """
    Fetch Arbitrum block numbers at num_days daily intervals starting from start. 
    In case of failed calls, retry a few times with exponential backoff.
    """
    blocks = []
    current_time = start
    for _ in range(num_days):
        retries = 3
        while retries > 0:
            try:
                block_number = get_block_number_by_time(apikey, current_time)
                blocks.append(block_number)
                break
            except ValueError as e:
                print(f"Error fetching block number: {e}. Retrying...")
                retries -= 1
                time.sleep(2 ** (3 - retries))  # Exponential backoff
        else:
            raise RuntimeError(f"Failed to fetch block number for {current_time} after multiple retries.")
        current_time += timedelta(days=1)
        time.sleep(0.3)
    return blocks


# Fetching blockchain data
DEPLOYMENTS = "./protocol/deployments/arbitrumMainnet"
MINTER_DEPLOYMENT_JSON = os.path.join(DEPLOYMENTS, "Minter.json")
BONDING_MANAGER_DEPLOYMENT_JSON = os.path.join(DEPLOYMENTS, "BondingManager.json")
BONDING_MANAGER_IMPLEMENTATION_JSON = os.path.join(DEPLOYMENTS, "BondingManagerTarget.json")

def arbitrum_w3():
    "Create default Web3 object from Arbitrum RPC URL."
    arb_rpc_url = os.getenv("ARB_RPC_URL")
    return Web3(Web3.HTTPProvider(arb_rpc_url))

def load_contract_from_json(w3, path):
    "Construct Contract object by parsing JSON loaded from path."
    with open(path, 'r') as file:
        contract_json = json.load(file)
    contract_abi = contract_json['abi']
    contract_address = contract_json['address']
    return w3.eth.contract(address=contract_address, abi=contract_abi)

def bonding_manager(w3):
    with open(BONDING_MANAGER_IMPLEMENTATION_JSON) as h:
        contract_abi = json.load(h)['abi']
    with open(BONDING_MANAGER_DEPLOYMENT_JSON) as h:
        contract_address = json.load(h)['address']
    return w3.eth.contract(address=contract_address, abi=contract_abi)
    

def fetch_historic(callable, blocks: list[int]) -> list[int]:
    results = []
    for block in blocks:
        results.append(callable.call(block_identifier=block))
        time.sleep(0.05)
    return results

def help() -> list[str]:
    return [
        "Set environment variables:",
        "ARBISCAN_API_KEY=<Arbiscan API key>\t\t(for fetching Arbitrum block numbers)",
        "ARB_RPC_URL=<Arbitrum archive node RPC URL>\t(for fetching historic state)"
    ]

if __name__ == "__main__":
    arbiscan_api_key = os.getenv("ARBISCAN_API_KEY")
    if not arbiscan_api_key:
        print(*help(), sep='\n')
        sys.exit(1)

    print("Fetching Arbitrum block numbers...")
    block_nums: list[int] = fetch_arbitrum_daily_blocks(apikey = arbiscan_api_key, start=datetime(2024, 1, 1, 0, 0, tzinfo=UTC), num_days=366)

    #with open("./data/arbitrum-daily-blocks-2024.json") as h:
    #    block_nums = json.load(h)


    w3 = arbitrum_w3()

    minter = load_contract_from_json(w3, MINTER_DEPLOYMENT_JSON)
    bonding = bonding_manager(w3)

    callables = {
        "inflation": minter.functions.inflation(),
        "total-supply": minter.functions.getGlobalTotalSupply(),
        "bonded": bonding.functions.getTotalBonded()
    }

    print("Fetching historic data...")
    results = {k: fetch_historic(v, block_nums) for k, v in callables.items()}

    print(json.dumps(results))
