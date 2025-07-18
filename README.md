# Bitcoin price analysis

## Getting started

- Initialise

```bash
python3 -m venv .venv
```

---

- Activate

```bash
source .venv/bin/activate
```

---

- Install dependencies

```bash
pip3 install -r requirements.txt
```

---

- Generate model

```bash
python3 gen.py
```

NOTE: this wil fail on the first run, fix the following manually.

- the BTCUSD_1H.csv chart needs to have the url removed from the first line of the csv.
- the BTC_sentiment.csv will be wrapped in json and the columns need reordering (date is first column and it has it as the last)

---

- Run data collection

```bash
python3 collect.py
```

To run `collect.py` in the background as a persistent process (so it doesn't stop when you close your terminal):

```bash
nohup ./.venv/bin/python3 collect.py &
```

This command uses `nohup` to prevent the process from being terminated when the controlling terminal is closed, and `&` to run it in the background. Output will be redirected to `nohup.out` by default.

This script connects to the Binance WebSocket stream to collect real-time BTCUSDT data and calculate features, logging everything to `BTCUSD_trading.csv`.

To stop the `collect.py` process if it was run with `nohup`:

1.  **Stopping the collect process:**

```bash
pgrep -f "python3 collect.py" | xargs kill
```

This command finds the PID of the running `collect.py` script and kills it in one step.

---

- Run trading bot

```bash
python3 trade.py
```

This script reads the `BTCUSD_trading.csv` file, loads the pre-trained model, makes price predictions, and logs trade actions to `BTCUSD_predictions_trades.csv`.

## Running Tests

To run the tests for this project, use the following command:

```bash
python3 -m unittest test_features.py
```

## Research

[] Gradient boosting machines (XGBoost, LightGBM or CatBoost), these models are good for regression tasks,
predicting future prices, or understanding feature importance.

[] Deep learning (LSTM or GRUs), specialised for sequential data. Great for capturing long-term dependencies
and complex patterns in price data. Particularly useful with large datasets.

[] Recurrent neural nets (temporal data, RNNs and variations). Designed for handling time series data and sequential dependencies
effectively.

[] GANS (Generative Adversarial Networks) can be used to simulate and generate synthetic price data for testing
trading strategies and risk analysis.

[] Bayesian machine learning - can help in probabalistic modeling, providing a range of possible outcomes and
quantifying uncertainty in predictions. Especially good for volatile assets like Bitcoin.

## Resources

- [machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading/).
