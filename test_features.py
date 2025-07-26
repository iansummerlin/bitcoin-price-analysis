
import unittest
import pandas as pd
from collect import prepare_exog_data
from unittest.mock import patch

class TestFeatureCalculation(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with enough data to calculate features.
        # This ensures that lookback periods for features (e.g., 120-period moving averages)
        # can be fully calculated for the new data point.
        num_initial_rows = 200 # Increased to provide ample historical data
        data = {
            'date': pd.to_datetime([f'2024-01-01 {i // 60:02d}:{i % 60:02d}:00' for i in range(num_initial_rows)]),
            'open': [100 + i for i in range(num_initial_rows)],
            'high': [110 + i for i in range(num_initial_rows)],
            'low': [90 + i for i in range(num_initial_rows)],
            'close': [105 + i for i in range(num_initial_rows)],
            'Volume BTC': [10 + i for i in range(num_initial_rows)],
            'Volume USD': [1050 + i * 10 for i in range(num_initial_rows)],
            'fng_value': [50 + i % 10 for i in range(num_initial_rows)]
        }
        self.df = pd.DataFrame(data)

    def test_feature_calculation_on_new_data(self):
        # Simulate a new data stream, representing a new incoming row of data.
        stream_data = {
            'E': 1704105600000,  # Example timestamp for 2024-01-01 12:00:00
            'o': 125,
            'h': 130,
            'l': 120,
            'c': 128,
            'v': 20,
            'q': 2560
        }
        mock_fng_value = 65 # Mock Fear & Greed Index value

        # Keep a copy of the DataFrame *before* the new data is added and features are calculated.
        # This allows us to verify that the original data remains untouched.
        df_before = self.df.copy()
        original_len = len(df_before)

        # Call the function under test: prepare_exog_data, which should append the new row
        # and calculate features for the updated DataFrame.
        df_after = prepare_exog_data(self.df, stream_data, mock_fng_value)

        # --- Assertions to ensure robustness and correctness ---

        # 1. Verify that the DataFrame length increased by exactly one.
        # This confirms that a single new row was appended.
        self.assertEqual(len(df_after), original_len + 1, "DataFrame length should increase by exactly one after adding new data.")

        # 2. Verify that the original data (all rows except the last one) remains unchanged.
        # This addresses the user's concern about "updating every single item in the data frame".
        # We compare the columns that existed in the original DataFrame (`df_before`)
        # with the corresponding columns in `df_after` (excluding the newly added last row).
        # This allows `prepare_exog_data` to add new feature columns to existing rows,
        # but ensures that the raw input data (e.g., 'open', 'high', 'close') for those
        # original rows is not altered.
        original_columns = df_before.columns.tolist()
        for col in original_columns:
            pd.testing.assert_series_equal(
                df_after.iloc[:-1][col].reset_index(drop=True), # Get original rows from df_after, reset index for comparison
                df_before[col].reset_index(drop=True),         # Get original rows from df_before, reset index for comparison
                check_dtype=False, # Allow for potential type changes (e.g., int to float) during processing
                check_names=False # Ignore Series name differences
            )

        # 3. Verify that the newly added last row has features calculated (not NaN).
        # This confirms that the feature engineering process successfully populated
        # the expected feature columns for the latest data point.
        new_row = df_after.iloc[-1]
        
        # List of expected feature columns that should be calculated by prepare_exog_data.
        expected_feature_columns = [
            'close_lag_90', 'close_lag_120', 'MA_7', 'MA_24', 'rsi', 
            'volatility_24', 'volatility_ewma_24', 'parkinson_volatility', 'atr_24'
        ]
        
        for col in expected_feature_columns:
            self.assertFalse(pd.isna(new_row[col]), f"Feature column '{col}' in new row should not be NaN.")
            # Optional: For even greater robustness, add specific assertions for the *values*
            # of key features in the new row, if you know their expected calculations.
            # Example:
            # if col == 'MA_7':
            #     # Calculate expected MA_7 based on the last 7 'close' prices from df_after
            #     expected_ma_7 = df_after['close'].iloc[-7:].mean()
            #     self.assertAlmostEqual(new_row[col], expected_ma_7, places=2, msg=f"MA_7 calculation incorrect for new row.")

        # 4. Verify that 'predicted_closing_price' column is NOT present.
        # This ensures that the feature calculation step does not prematurely add
        # prediction-related columns.
        self.assertNotIn('predicted_closing_price', df_after.columns, "'predicted_closing_price' should not be present after feature calculation.")

if __name__ == '__main__':
    unittest.main()
