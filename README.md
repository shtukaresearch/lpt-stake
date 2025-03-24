First run `fetch_data.py` to collect historic daily data from 2024 and dump to terminal.

```
mkdir data
uv run fetch_data.py > ./data/lpt-data.json
```

Historic BTC prices can be downloaded here: https://www.coingecko.com/en/coins/bitcoin/historical_data
Download in CSV format and save in the `./data/` folder (with default filename `btc-usd-max.csv`).

Then run notebook `lpt-eda.py` using Marimo:

```
uv run marimo edit lpt-eda.py
```
