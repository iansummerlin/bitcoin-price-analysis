# Future Improvements for BTC Hourly Closing Price Model

## 1. Current Model Analysis

The current model utilizes an ARIMA(p, d, q) architecture with several exogenous variables to predict the hourly closing price of Bitcoin. The `CHANGELOG.txt` indicates a continuous effort to improve the model's performance, primarily measured by RMSE and MAE.

**Key Observations from `gen.py` and `CHANGELOG.txt`:**

*   **Model Type:** ARIMA (Autoregressive Integrated Moving Average) with exogenous variables.
*   **Features Used:** `close_lag_90`, `close_lag_120`, `MA_7`, `MA_24`, `rsi`, `volatility_24`, `volatility_ewma_24`, `parkinson_volatility`, `atr_24`, `fng_value`.
*   **Performance:** The most recent entry (18/07/2025) shows RMSE: 503.37 and MAE: 328.24 with an ARIMA(4, 1, 5) model. This is an improvement from earlier versions, notably after implementing grid search for ARIMA parameters (30/11/2024, RMSE: 347.90, MAE: 218.56).
*   **Feature Effectiveness:**
    *   `rsi`, `volatility_ewma_24`, `parkinson_volatility`, `atr_24`, and `fng_value` have shown positive impacts on reducing errors.
    *   `sharpe_ratio` and `fng_classification_encoded` were tested but found to be ineffective (high p-values or increased errors) and subsequently removed or not used.
    *   `close_lag_24` and `close_lag_250` were also found to worsen the model.
*   **Statistical Significance (p-values):** Some current features, like `ar.L1` (0.350) in the latest model and `volatility_ewma_24` (0.103) in the 30/11/2024 model, show high p-values, suggesting they might not be statistically significant predictors.
*   **Heteroskedasticity:** The `CHANGELOG.txt` consistently shows high Heteroskedasticity (H) values with Prob(H) (two-sided) at 0.00, indicating the presence of heteroskedasticity (non-constant variance of errors). This is also noted as a minor improvement in `gen.py`.
*   **Kurtosis:** High Kurtosis values (e.g., 33.44 in the latest model) suggest heavy tails in the error distribution, implying more extreme outliers than a normal distribution.
*   **Model Complexity:** The `gen.py` notes suggest exploring more features (10-20) and advanced models like LSTM/GRU.

## 2. Proposed Improvements

Based on the analysis, here are detailed recommendations for improving the BTC hourly closing price model:

### 2.1. Feature Engineering & Selection

*   **Re-evaluate Existing Features:**
    *   **Statistical Significance:** Re-examine features with high p-values (e.g., `ar.L1`, `volatility_ewma_24`). Consider removing them or transforming them if they do not contribute significantly to the model's predictive power.
    *   **`parkinson_volatility`:** Investigate the large negative coefficient of `parkinson_volatility` (-1543.3560 in the latest model). While statistically significant, such a large coefficient might indicate scaling issues or potential multicollinearity with other volatility measures. Consider standardizing features or exploring alternative volatility calculations.
*   **Explore New Features:**
    *   **Trading Volume:** Incorporate trading volume as an exogenous variable. High volume often accompanies significant price movements.
    *   **Order Book Data:** If accessible, features derived from order book depth, bid-ask spread, and order imbalances can provide valuable short-term insights.
    *   **On-Chain Data:** Explore on-chain metrics like active addresses, transaction count, or whale movements, which can indicate network activity and sentiment.
    *   **Macroeconomic Indicators:** While hourly data might be too granular, consider how broader economic indicators (e.g., interest rates, inflation news) could indirectly influence BTC price, perhaps as daily or weekly features.
    *   **Social Media Sentiment (beyond FNG):** Utilize sentiment analysis from platforms like Twitter or Reddit, as these can capture real-time market mood.
    *   **Google Trends Data:** Search interest for "Bitcoin" or related terms could serve as a proxy for public interest.
*   **Advanced Feature Engineering:**
    *   **Interaction Terms:** Explore interactions between existing features (e.g., `rsi` * `volatility_24`) to capture non-linear relationships.
    *   **Polynomial Features:** Introduce polynomial terms for certain features to model non-linear dependencies.
    *   **Lagged Exogenous Variables:** Experiment with lagged versions of exogenous variables, as their impact on price might not be immediate.
*   **Automated Feature Selection:** Implement techniques like Recursive Feature Elimination (RFE), Lasso Regression, or tree-based feature importance to automatically select the most relevant features and reduce model complexity.

### 2.2. Model Architecture & Refinement

*   **Address Heteroskedasticity:**
    *   **GARCH Models:** As noted in `gen.py`, explicitly model the volatility using GARCH (Generalized Autoregressive Conditional Heteroskedasticity) or GARCH-in-mean models. This will allow the model to account for the changing variance of errors, which is crucial for financial time series.
    *   **Transformations:** Consider applying transformations (e.g., logarithmic) to the `close` price if the variance of the series itself is non-constant.
*   **Explore Advanced Time Series Models:**
    *   **LSTM/GRU Networks:** Given the sequential nature of time series data and the potential for long-term dependencies, Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) neural networks are strong candidates. These models are particularly adept at learning patterns over extended sequences.
    *   **Prophet (Facebook):** For forecasting with trend, seasonality, and holidays, Prophet can be a robust and easy-to-use alternative, especially if strong seasonal patterns are observed.
    *   **Hybrid Models:** Combine the strengths of different models. For example, an ARIMA-LSTM hybrid could use ARIMA to capture linear dependencies and LSTM for non-linear patterns in the residuals.
*   **Ensemble Methods:** Combine predictions from multiple models (e.g., ARIMA, LSTM, XGBoost) using techniques like averaging, weighted averaging, or stacking. Ensembles often lead to more robust and accurate forecasts.
*   **Hyperparameter Optimization:**
    *   **More Extensive Grid Search:** Expand the search space for ARIMA (p, d, q) parameters.
    *   **Bayesian Optimization:** For more complex models or larger hyperparameter spaces, Bayesian optimization can efficiently find optimal hyperparameters by intelligently exploring the search space.

### 2.3. Data Considerations

*   **Data Frequency:** While the current model uses hourly data, consider if higher frequency data (e.g., 15-minute or 5-minute) could capture more rapid price dynamics, albeit with increased computational cost.
*   **Data Cleaning & Outlier Handling:** Implement more sophisticated outlier detection and handling mechanisms. The high kurtosis suggests the presence of significant outliers that might be skewing the model.
*   **Data Source Reliability:** Ensure the reliability and consistency of data sources for both price and sentiment data.

### 2.4. Evaluation & Robustness

*   **Walk-Forward Validation:** Instead of a single train-test split, implement walk-forward validation (or rolling-origin evaluation). This simulates real-world forecasting by retraining the model periodically on expanding or rolling windows of data, providing a more realistic assessment of performance.
*   **Additional Metrics:**
    *   **Directional Accuracy:** For trading applications, evaluate the model's ability to correctly predict the direction of price movement (up/down).
    *   **Sharpe Ratio/Profitability (if applicable):** If the model is intended for trading, backtest a simple trading strategy based on its predictions and evaluate its profitability and risk-adjusted returns (e.g., Sharpe Ratio).
*   **Residual Analysis:** Conduct thorough analysis of model residuals (e.g., autocorrelation plots, normality tests) to ensure they are white noise and meet model assumptions.

### 2.5. Operational Improvements

*   **Automated Data Ingestion:** Ensure a robust and automated pipeline for ingesting the latest price and sentiment data.
*   **MLOps Pipeline:** Implement a comprehensive MLOps pipeline for continuous integration, continuous training, and continuous deployment of the model. This includes automated retraining, monitoring performance in production, and alerting for degradation.
*   **Merged CSV:** As noted in `gen.py`, pre-generating and downloading the merged CSV will significantly speed up model generation runs.

## 3. Conclusion

The current ARIMA model provides a solid foundation for BTC hourly price prediction. The next steps should focus on addressing the identified statistical issues (heteroskedasticity, non-significant features), exploring more advanced time series models (LSTMs, GARCH), and enriching the feature set with more diverse and impactful indicators. Implementing robust evaluation techniques like walk-forward validation and building an MLOps pipeline will ensure the model remains accurate and reliable in a dynamic market environment.
