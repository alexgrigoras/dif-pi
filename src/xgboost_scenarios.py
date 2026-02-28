import xgboost as xgb
import numpy as np
import pandas as pd

class XGBoostScenarioGeneration:
    def __init__(self):
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)

    def train(self, data, input_col, output_col):
        """Train XGBoost on historical data"""
        self.input_col = input_col
        self.output_col = output_col

        X = np.log(data[[input_col]])
        y = np.log(data[output_col])
        self.model.fit(X, y)

    def generate_scenarios(self, data, price_change_percentages, forecast_length=7):
        """Generate future what-if scenarios for forecast_length steps"""
        scenarios_input = {}
        scenarios_output = {}

        last_price = data[self.input_col].iloc[-1]
        last_demand = data[self.output_col].iloc[-1]
        last_index = data.index[-1]

        future_index = pd.date_range(start=last_index + pd.DateOffset(months=1), periods=forecast_length, freq='M')

        for change in price_change_percentages:
            scenario_name = f'{self.input_col} change {change}%'

            adjusted_price = np.full(forecast_length, last_price * (1 + change / 100))
            X = np.log(adjusted_price).reshape(-1, 1)
            pred_log_demand = self.model.predict(X)
            pred_demand = np.exp(pred_log_demand)

            full_index = [last_index] + list(future_index)
            full_prices = np.concatenate([[last_price], adjusted_price])
            full_demands = np.concatenate([[last_demand], pred_demand])

            scenarios_input[scenario_name] = pd.DataFrame({self.input_col: full_prices}, index=full_index)
            scenarios_output[scenario_name] = pd.DataFrame({self.output_col: full_demands}, index=full_index)

        return scenarios_input, scenarios_output