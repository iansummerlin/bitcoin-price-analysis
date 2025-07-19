# Gemini Safety Instructions

To prevent data loss and ensure the integrity of this project, I must adhere to the following rules, especially when creating or modifying tests:

1.  **No Teardowns for Critical Data Files:** I will not write any test code that deletes, modifies, or includes teardown steps for critical data files.

2.  **Protected Files:** This rule applies specifically to the following files, but also to any other files that appear to contain important generated data:
    *   `BTCUSD_trading.csv` (Primary data collection file)
    *   `BTCUSD_predictions_trades.csv` (Trading log file)
    *   `arima-btc-closing-price.pkl` (and any other `.pkl` model files)

My primary directive is to preserve these assets. I will avoid any operations that put them at risk.
