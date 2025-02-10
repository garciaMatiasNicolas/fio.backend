import numpy as np
import pandas as pd


class Error:

    def __init__(self, error_method: str, error_periods: int, actual, predicted):
        self.error_method = error_method
        self.error_periods = error_periods
        self.actual = actual
        self.predicted = predicted

        if self.error_method not in ['MAE', 'MAPE', 'SMAPE', 'RMSE']:
            raise ValueError("Invalid error_method. Use 'MAE', 'MAPE', 'SMAPE', 'RMSE'.")

    @staticmethod
    def calculate_mape(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            mape = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            mape = 100
        elif predicted < 0:
            mape = 100
        else:
            mape = abs((actual - predicted) / actual) * 100

        return round(float(mape), 2)

    @staticmethod
    def calculate_rmse(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            rmse = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            rmse = 100
        elif predicted < 0:
            rmse = 100
        else:
            rmse = (actual - predicted) ** 2

        return round(np.sqrt(rmse), 2)

    @staticmethod
    def calculate_mae(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            mae = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            mae = 100
        elif predicted < 0:
            mae = 100
        else:
            mae = abs(actual - predicted)

        return round(mae, 2)

    @staticmethod
    def calculate_smape(actual: float, predicted: float) -> float:
        if actual == 0 and predicted == 0:
            smape = 0
        elif (actual == 0 or actual < 0) and predicted != 0:
            smape = 100
        elif predicted < 0:
            smape = 100
        else:
            smape = abs((actual - predicted) / ((actual + predicted) / 2)) * 100

        return round(smape, 2)

    def calculate_error_periods_selected(self):

        if self.error_periods != 0:
            predicted = self.predicted[-self.error_periods:]
            actual = self.actual[-self.error_periods:]

        else:
            predicted = self.predicted
            actual = self.actual

        error_values = 0
        last_period_error = 0
        error_abs = 0

        if self.error_method == "MAPE":
            error_values = [self.calculate_mape(actual, predicted) for actual, predicted in zip(actual, predicted)]
            last_period_error = self.calculate_mape(predicted=predicted[-1], actual=actual[-1])

        if self.error_method == "SMAPE":
            error_values = [self.calculate_smape(actual, predicted) for actual, predicted in zip(actual, predicted)]
            last_period_error = self.calculate_smape(predicted=predicted[-1], actual=actual[-1])

        if self.error_method == "MAE":
            error_values = [self.calculate_mae(actual, predicted) for actual, predicted in zip(actual, predicted)]
            last_period_error = self.calculate_mae(predicted=predicted[-1], actual=actual[-1])

        if self.error_method == "RMSE":
            error_values = [self.calculate_rmse(actual, predicted) for actual, predicted in zip(actual, predicted)]
            last_period_error = self.calculate_rmse(predicted=predicted[-1], actual=actual[-1])

        error = round(np.mean(error_values), 2)

        return error, round(last_period_error, 2)

    def calculate_error_last_period(self, prediction_periods: int, df: pd.DataFrame) -> tuple[float, float]:
        methods = {
            'MAPE': Error.calculate_mape,
            'SMAPE': Error.calculate_smape,
            'RMSE': Error.calculate_rmse,
            'MAE': Error.calculate_mae
        }
        

        last_period_column = prediction_periods + 2
        last_period = df.iloc[:, -last_period_column]

        values = []
        actual_vals = []
        predicted_vals = []

        for i in range(0, len(last_period), 2):
            actual = last_period[i]
            predicted = last_period[i + 1]
            actual_vals.append(actual)
            predicted_vals.append(predicted)
            error = 0

            if self.error_method in methods:
                calc_error = methods[self.error_method]
                error = calc_error(predicted, actual)

            values.append(error)

        absolute_error = 0
        if self.error_method == 'MAPE':
            absolute_error = self.calculate_mape(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'SMAPE':
            absolute_error = self.calculate_smape(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'MAE':
            absolute_error = self.calculate_mae(predicted=sum(predicted_vals), actual=sum(actual_vals))

        if self.error_method == 'RMSE':
            absolute_error = self.calculate_rmse(predicted=sum(predicted_vals), actual=sum(actual_vals))

        return absolute_error

