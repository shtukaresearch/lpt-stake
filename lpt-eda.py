import marimo

__generated_with = "0.11.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import scipy
    import datetime
    from matplotlib import pyplot as plt
    import json
    return datetime, json, mpl, np, pd, plt, scipy


@app.cell
def _():
    START_DATE = '2024-01-01'
    END_DATE   = '2024-12-31'
    return END_DATE, START_DATE


@app.cell
def _(END_DATE, START_DATE, json, pd):
    with open("./data/lpt-data.json") as h:
        lpt_data = json.load(h)

    lpt_data['bonded-rate'] = [lpt_data['bonded'][i] / lpt_data['total-supply'][i] for i in range(len(lpt_data['bonded']))]

    date_index = pd.date_range(start=START_DATE, end=END_DATE, tz="UTC")

    bonded_series = pd.Series(lpt_data['bonded-rate'], index=date_index)
    issuance_series = pd.Series(lpt_data['inflation'], index=date_index)

    print(f"Date of peak bonded rate: {date_index[bonded_series.argmax()]}")
    return bonded_series, date_index, h, issuance_series, lpt_data


@app.cell
def _():
    # SPX data from NASDAQ
    # https://www.nasdaq.com/market-activity/index/spx/historical
    # Fetch the dataset and remove quotes to play with SPX as a regressand

    """
    spx_df = pd.read_csv("./data/HistoricalData_1741932948522.csv")
    spx_df.drop(columns=['Open', 'High', 'Low'], inplace=True)

    spx_df["30MA"] = spx_df["Close/Last"].rolling(30).mean()

    spx_df["Date"] = pd.to_datetime(spx_df["Date"], utc=True)
    spx_df.sort_values(by="Date", inplace=True)
    spx_df = spx_df[spx_df["Date"].between(START_DATE, END_DATE)]
    spx_df.set_index("Date", inplace=True)

    spx_df['30MA'].plot(title="SPX 30-day moving average")
    """
    return


@app.cell
def _(END_DATE, START_DATE, pd):
    btc_df = pd.read_csv("./data/btc-usd-max.csv") # columns = ['snapped_at', 'price', 'market_cap', 'total_volume']
    btc_df.drop(columns=['market_cap', 'total_volume'], inplace=True)
    btc_df['30MA'] = btc_df['price'].rolling(30).mean()
    btc_df['ROI'] = btc_df['price'].pct_change(180)

    btc_df["Date"] = pd.to_datetime(btc_df["snapped_at"])
    btc_df.sort_values(by="Date", inplace=True)
    btc_df = btc_df[btc_df["Date"].between(START_DATE, END_DATE)]

    btc_df.set_index("Date", inplace=True)
    return (btc_df,)


@app.cell
def _(bonded_series, btc_df, issuance_series, pd):
    market_df = pd.DataFrame(index=btc_df.index)
    market_df['bonded'] = bonded_series
    market_df['issuance'] = issuance_series
    market_df['btc'] = btc_df['30MA']
    market_df
    return (market_df,)


@app.cell
def _(market_df):
    # Covariance and correlations
    market_df.corr()
    return


@app.cell
def _(bonded_series, issuance_series, plt):
    # Scatter plot of issuance against bonded
    plt.scatter(bonded_series, issuance_series)
    return


@app.cell
def _(market_df, plt):
    plt.scatter(market_df['bonded'], market_df["btc"])
    return


@app.cell
def _(market_df, np, pd):
    import statsmodels.api as sm

    def normalize(s):
        return (s - s.mean()) / s.std()

    # Independent variables (X) and dependent variable (Y)
    X = pd.DataFrame({
        #"spx": normalize(np.log(market_df['30MA'])), 
        "btc": normalize(np.log(market_df['btc'])),
        "iss": normalize(np.log(market_df['issuance']  ))
        }, index=market_df.index)
    Y = normalize(np.log(market_df['bonded']))

    # No constant needed because data is centred
    #X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(Y, X)
    assert (model.exog == X.values).all() # Input data saved as model.exog

    results = model.fit()
    assert (results.predict() == X @ results.params).all() # Linear model predictions

    # Summary of the regression
    print(results.summary())
    return X, Y, model, normalize, results, sm


@app.cell
def _(results):
    print(results.params)
    print(results.cov_params())
    return


@app.cell
def _(plt, results):
    # Residuals plot
    plt.scatter(results.predict(), results.resid)
    plt.gca().set_title("Residuals")
    plt.gca().set_xlabel("Prediction")
    plt.gca().set_ylabel("Residual")
    plt.savefig("residuals.svg")
    plt.show()
    return


@app.cell
def _(Y, plt, results):
    fig, axs = plt.subplots(3,1, figsize=(15,12))

    Y.plot(ax=axs[0], title="data")
    res = results.get_prediction().summary_frame(alpha=0.05)
    res["mean"].plot(ax=axs[1], title="predictions", color="purple")
    results.resid.plot(ax=axs[2], title="residuals", color="red")

    plt.tight_layout()

    plt.savefig("series.svg")

    plt.show()
    return axs, fig, res


@app.cell
def _(datetime, market_df, np, results):
    ROUNDLENGTH = datetime.timedelta(seconds=6377*12)
    ROUNDS_PER_YEAR = datetime.timedelta(days=365) / ROUNDLENGTH

    def apy_to_issuance(apy: float):
        rpy = (1 + apy) ** (1/ROUNDS_PER_YEAR) - 1
        return int(rpy * 1_000_000_000)

    def issuance_to_apy(iss: int):
        rpy = iss / 1_000_000_000
        return (1 + rpy) ** ROUNDS_PER_YEAR - 1

    btc = np.log(market_df['btc'])
    iss = np.log(market_df['issuance'])
    bond = np.log(market_df['bonded'])

    def print_rates(param, btcusd):
        test_inputs = [
            (np.log(btcusd)-btc.mean())/btc.std(), 
            (np.log(param)-iss.mean())/iss.std() 
        ]
        rate = np.exp(results.predict(test_inputs) * bond.std() + bond.mean())[0]

        print(f"Equilibrium rate: {rate}")
        print(f"APY @ equilibrium: {issuance_to_apy(param)/rate}")
    return (
        ROUNDLENGTH,
        ROUNDS_PER_YEAR,
        apy_to_issuance,
        bond,
        btc,
        iss,
        issuance_to_apy,
        print_rates,
    )


@app.cell
def _(print_rates):
    param_grid = [
        (900_000, 80_000),
        (900_000, 100_000),
        (750_000, 80_000),
        (750_000, 100_000),
        (250_000, 80_000),
        (250_000, 100_000),
    ]

    for parms in param_grid:
        print_rates(*parms)
    return param_grid, parms


@app.cell
def _(apy_to_issuance):
    (apy_to_issuance(0.3) - apy_to_issuance(0.2)) / 500
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
