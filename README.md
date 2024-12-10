# Bitcoin price analysis

## Getting started

- If using venv

Initialise

```bash
python3 venv -m ~/.venv
```

Activate

```bash
source ~/.venv/bin/activate
```

- Generate model

```bash
python3 gen.py
```

- Run model

```bash
python3 main.py
```

NOTE: the first time downloading the csv data you will need to do some manual formatting. Maybe i'll automate this step one day.

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
