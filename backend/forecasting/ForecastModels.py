import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import Lasso, LinearRegression, BayesianRidge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from prophet import Prophet
from neuralprophet import NeuralProphet
import numpy as np
from autogluon.timeseries import TimeSeriesPredictor
import warnings
warnings.filterwarnings("ignore")

class ForecastModels:

    def autogluon(
        idx: int, 
        row: list, 
        prediction_periods: int, 
        dates: list, 
        explosive: float,
        model: str
    ):
        # Validaciones
        if len(dates) != len(row):
            raise ValueError(f"Las longitudes de las fechas y los valores no coinciden para el índice {idx}")

        df = pd.DataFrame({'timestamp': dates, 'target': row})
        df['item_id'] = idx
        avg_historical = df['target'].mean()

        # Verificación de datos
        if df['target'].isnull().any():
            raise ValueError(f"Hay valores nulos en la columna 'target' para el índice {idx}")

        if df['target'].nunique() == 1:
            return idx, [df['target'].iloc[0]] * (len(row) + prediction_periods)

        # Hiperparámetros ajustados para mayor variabilidad
        hyperparameters = {
            model: {
                "learning_rate": 0.5,  # Reducimos la tasa de aprendizaje
                "num_layers": 50,  # Menos capas para mayor flexibilidad
                "hidden_size": 512,  # Reducimos el tamaño de las capas ocultas
                "dropout_rate": 0.3,  # Evitamos sobreajuste pero no demasiado alto
                "early_stopping": True
            }
        }
        # Entrenar modelo
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_periods,
            eval_metric="MAPE"  # Se usa MAPE como métrica de error
        )

        predictor.fit(
            df,
            hyperparameters=hyperparameters,
            time_limit=10,
            verbosity=2
        )

        # Obtener predicciones futuras
        future_predictions = predictor.predict(df)['mean']

        # Asegurar que el tamaño de las predicciones es correcto
        if len(future_predictions) != prediction_periods:
            raise ValueError(
                f"El tamaño de las predicciones futuras es incorrecto. Esperado: {prediction_periods}, obtenido: {len(future_predictions)}"
            )

        # Agregar ruido aleatorio para simular variabilidad
        noise = np.random.normal(scale=avg_historical * 0.05, size=prediction_periods)
        future_predictions = [max(pred + n, 0) for pred, n in zip(future_predictions, noise)]

        # Combinar predicciones históricas con futuras
        total_predictions = list(row) + list(future_predictions)

        # Aplicar límite en caso de evento explosivo
        if explosive > 0.0:
            total_predictions = [min(pred, avg_historical * explosive) for pred in total_predictions]

        return idx, total_predictions

    @staticmethod
    def arimax(
        idx: int, 
        row: list, 
        test_periods: int, 
        prediction_periods: int, 
        additional_params: dict, 
        explosive: float, 
        exog_data: pd.DataFrame = None
    ):
        ## ARIMA PARAMS ## 
        if additional_params is not None and 'arima_params' in additional_params:
            p, d, q = additional_params['arima_params']
            arima_order = (int(p), int(d), int(q))
        else:
            arima_order = (1, 1, 0) 

        # Convertir la serie temporal en un objeto pandas Series
        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()

        # Dividir los datos en entrenamiento y prueba
        train_data = time_series[:-test_periods]
        n_train = len(train_data)

        # Preparar variables exógenas (CORREGIDO)
        if exog_data is not None:
            exog_data['date'] = pd.to_datetime(exog_data['date']) # Asegurar tipo datetime
            exog_data = exog_data.set_index('date') # Establecer 'DATE' como índice

            exog_train = exog_data.iloc[:n_train]['exog_value'] # Seleccionar 'EXOG' y usar slicing con .iloc
            exog_test = exog_data.iloc[n_train:n_train + test_periods]['exog_value']
            exog_future = exog_data.iloc[n_train + test_periods:n_train + test_periods + prediction_periods]['exog_value']
        else:
            exog_train = exog_test = exog_future = None

        # Ajustar el modelo ARIMAX
        model = ARIMA(train_data, order=arima_order, exog=exog_train)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()

        # Predicciones
        train_predictions = model_fit.predict(start=0, end=n_train - 1, exog=exog_train)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1, exog=exog_test)
        future_predictions = model_fit.forecast(steps=prediction_periods, exog=exog_future)

        # Consolidar todas las predicciones
        total_predictions = (
            list(train_predictions.values) +
            list(test_predictions.values) +
            list(future_predictions.values)
        )

        # Ajustar valores negativos y explosivos
        total_predictions = [max(prediction, 0) for prediction in total_predictions]
        if explosive > 0.0:
            total_predictions = [
                min(prediction, avg_historical * explosive) for prediction in total_predictions
            ]

        return idx, total_predictions

    @staticmethod
    def sarimax(idx, row, test_periods, prediction_periods, additional_params, explosive, exog_data):

        # Validar parámetros SARIMA
        if additional_params is not None and 'sarima_params' in additional_params:
            p, d, q = additional_params['sarima_params']
            seasonal_order = additional_params.get('seasonal_params', (0, 0, 0, 0))
        else:
            # Valores predeterminados si no se proporcionan parámetros
            p, d, q = 1, 1, 0
            seasonal_order = (1, 0, 0, 12)

        sarima_order = (int(p), int(d), int(q))
        seasonal_order = tuple(map(int, seasonal_order))

        # Convertir la serie temporal en un objeto pandas Series
        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()

        # Dividir los datos en entrenamiento y prueba
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)

        # Preparar variables exógenas
        if exog_data is not None:
            exog_train = exog_data.iloc[:n_train]
            exog_test = exog_data.iloc[n_train:n_train + test_periods]
            exog_future = exog_data.iloc[n_train + test_periods:n_train + test_periods + prediction_periods]
        else:
            exog_train = exog_test = exog_future = None

        # Ajustar el modelo SARIMAX
        model = SARIMAX(train_data, order=sarima_order, seasonal_order=seasonal_order, exog=exog_train)
        model_fit = model.fit(disp=False)

        # Predicciones
        train_predictions = model_fit.predict(start=0, end=n_train - 1, exog=exog_train)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1, exog=exog_test)
        future_predictions = model_fit.forecast(steps=prediction_periods, exog=exog_future)

        # Consolidar todas las predicciones
        total_predictions = (
                list(train_predictions.values) +
                list(test_predictions.values) +
                list(future_predictions.values)
        )

        # Ajustar valores negativos y explosivos
        total_predictions = [max(prediction, 0) for prediction in total_predictions]
        if explosive > 0.0:
            total_predictions = [
                min(prediction, avg_historical * explosive) for prediction in total_predictions
            ]

        return idx, total_predictions

    def prophet_holidays(
        idx: int, 
        row: list, 
        prediction_periods: int, 
        exog_data: pd.DataFrame, 
        dates: list, 
        explosive: float, 
        detect_outliers: bool = True,
        monthly: bool = True
    ):
        ## PROPHET PARAMS AND HISTORICAL DATA ##
        seasonality_mode = "additive"
        seasonality_prior_scale = 1
        uncertainty_samples = 1000
        changepoint_prior_scale = 0.001
        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': row})

        ## DETECT OUTLIERS ##
        def detect_outliers_func(series, threshold=3):
            mean = series.mean()
            std_dev = series.std()
            z_scores = (series - mean) / std_dev
            return z_scores.abs() > threshold

        if detect_outliers:
            outliers = detect_outliers_func(df['y'])
            df['outliers'] = outliers

        ## EXOGENOUS VALUES ##
        exog = exog_data.rename(columns={'date': 'ds', 'variable': 'holiday'})
        exog.drop(columns=['product_id', 'exog_value'], inplace=True)
        exog['lower_window'] = 0.0
        exog['upper_window'] = 30.0

        ## MODEL AND PREDICTIONS ##
        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=12,
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=uncertainty_samples,
            holidays=exog
        )

        avg_historical = df['y'].mean()

        if detect_outliers:
            model.fit(df[~df['outliers']])
        
        else:
            model.fit(df)
        
        future = model.make_future_dataframe(periods=prediction_periods, freq="MS" if monthly else "D")
        forecast = model.predict(future)
        total_predictions = forecast['yhat'].to_list()

        total_predictions = [max(0, prediction) for prediction in total_predictions]

        if explosive > 0.0:
            total_predictions = [
                min(prediction, avg_historical * explosive) for prediction in total_predictions
            ]

        return idx, total_predictions

    def prophet_exog(
        idx: int, 
        row: list, 
        prediction_periods: int, 
        exog_data: pd.DataFrame, 
        dates: list, 
        explosive: float, 
        projected_exog_data: pd.DataFrame = None, 
        monthly: bool = True
    ):
        ## PROPHET PARAMS AND HISTORICAL DATA ##
        seasonality_mode = "additive"
        seasonality_prior_scale = 1
        uncertainty_samples = 1000
        changepoint_prior_scale = 0.001

        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': row})
        avg_historical = df['y'].mean()

        ## MODEL AND PREDICTIONS ##
        model = Prophet(
            weekly_seasonality=False,
            yearly_seasonality=12,  
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=uncertainty_samples,
        )

        if monthly:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        if not exog_data.empty:
            exog_data.drop(columns=['product_id'], inplace=True)
            exog_data = exog_data.pivot(index='date', columns='variable', values='exog_value')
            exog_data.index = pd.to_datetime(exog_data.index)
            df = df.merge(exog_data, left_on='ds', right_index=True, how='left').fillna(0.0)
            
            exog_variables = list(exog_data.columns)
            for var in exog_variables:
                model.add_regressor(var, standardize=False, prior_scale=1, mode='multiplicative')

        model.fit(df)

        if projected_exog_data is not None and not projected_exog_data.empty:
            projected_exog_data = projected_exog_data.pivot(index='date', columns='variable', values='exog')
            projected_exog_data.index = pd.to_datetime(projected_exog_data.index)

            future = model.make_future_dataframe(periods=prediction_periods, freq="MS" if monthly else "D")
            future = future.merge(projected_exog_data, left_on='ds', right_index=True, how='left').fillna(0.0)
            
            for var in exog_variables:
                if var not in projected_exog_data.columns:
                    future[var] = [0.0] * len(future)
        else:
            future = model.make_future_dataframe(periods=prediction_periods, freq="MS" if monthly else "D")
            for var in exog_variables:
                future[var] = df[var]
            
            future.fillna(0.0, inplace=True)

        forecast = model.predict(future)
        total_predictions = forecast['yhat'].to_list()

        ## OTHER VALIDATIONS ##
        total_predictions = [max(0, prediction) for prediction in total_predictions]

        if explosive > 0.0:
            total_predictions = [
                min(prediction, avg_historical * explosive) for prediction in total_predictions
            ]

        return idx, total_predictions

    @staticmethod
    def holt_holtwinters_ema(idx, row, test_periods, prediction_periods, model_name, seasonal_periods, is_daily, explosive):
        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()

        model = ''

        if model_name == 'holtsWintersExponentialSmoothing':
            try:
                seasonal_periods = int(seasonal_periods)
            except:
                seasonal_periods = float(seasonal_periods)
                seasonal_periods = int(seasonal_periods)

            if is_daily:
                model = ExponentialSmoothing(time_series, trend='add')

            else:
                model = ExponentialSmoothing(time_series, seasonal_periods=int(seasonal_periods), trend='add', seasonal='add') 

        if model_name == 'holtsExponentialSmoothing':
            model = ExponentialSmoothing(time_series, trend='add')

        if model_name == 'exponential_moving_average':
            model = ExponentialSmoothing(time_series, trend=None, seasonal=None)

        model_fit = model.fit()

        train_pred = model_fit.predict(0)
        forecast = model_fit.forecast(prediction_periods)
        total_predictions = forecast.tolist() + train_pred.tolist()
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

        return idx, total_predictions

    @staticmethod
    def exponential_smoothing(idx, row, test_periods, prediction_periods, explosive):
        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        model = SimpleExpSmoothing(train_data)

        model_fit = model.fit()

        test_predictions = model_fit.forecast(test_periods)
        train_predictions = model_fit.fittedvalues
        smoothed_series = time_series.ewm(span=10, min_periods=0).mean()
        future_predictions = smoothed_series.iloc[-1:].repeat(prediction_periods)
        total_predictions = list(train_predictions.values) + list(test_predictions.values) + list(future_predictions.values)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]
        
        return idx, total_predictions

    @staticmethod
    def arima(idx, row, test_periods, prediction_periods, additional_params, explosive):

        if additional_params is not None:
            p, d, q = additional_params[f'arima_params']
            arima_order = (int(p), int(d), int(q))

        else:
            arima_order = (0, 0, 0)

        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)

        model = ARIMA(train_data, order=arima_order)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        train_predictions = model_fit.predict(start=0, end=n_train - 1)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)
        future_predictions = model_fit.forecast(steps=prediction_periods)

        total_predictions = list(train_predictions.values) + list(test_predictions.values) + list(future_predictions.values)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]

        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

        return idx, total_predictions

    @staticmethod
    def sarima(idx, row, test_periods, prediction_periods, additional_params, explosive):

        if additional_params is not None:
            p, d, q = additional_params[f'sarima_params']
            sarima_order = (int(p), int(d), int(q))

        else:
            sarima_order = (0, 0, 0)

        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]
        n_train = len(train_data)

        model = SARIMAX(train_data, order=sarima_order)
        model.initialize_approximate_diffuse()
        model_fit = model.fit()
        train_predictions = model_fit.predict(start=0, end=n_train - 1)
        test_predictions = model_fit.predict(start=n_train, end=len(time_series) - 1)
        future_predictions = model_fit.forecast(steps=prediction_periods)

        total_predictions = list(train_predictions.values) + list(test_predictions.values) + list(
            future_predictions.values)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]

        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

        return idx, total_predictions
    
    def prophet(
        idx: int, 
        row: list, 
        prediction_periods: int, 
        additional_params: dict, 
        seasonal_periods: int, 
        dates: list, 
        detect_outliers: bool, 
        explosive: float
    ):
        try:
            seasonal_periods = int(seasonal_periods)
        except:
            seasonal_periods = float(seasonal_periods)
            seasonal_periods = int(seasonal_periods)

        def detect_outliers_func(series, threshold=3):
            mean = series.mean()
            std_dev = series.std()
            z_scores = (series - mean) / std_dev
            return z_scores.abs() > threshold

        # Crear DataFrame
        df = pd.DataFrame({'ds': pd.to_datetime(dates), 'y': row})
        df['floor'] = 0
        avg_historical = df['y'].mean()
        max_cap = abs(avg_historical * 2)
        df['cap'] = max_cap

        # Detectar frecuencia de los datos (diarios o mensuales)
        freq = pd.infer_freq(df['ds'])
        if freq is None:
            raise ValueError("No se pudo inferir la frecuencia de los datos. Verifique las fechas proporcionadas.")
        freq_is_daily = freq.lower().startswith('d')

        # Configurar parámetros adicionales o valores predeterminados
        if additional_params is not None:
            seasonality_mode = additional_params[0]
            seasonality_prior_scale = float(additional_params[1])
            uncertainty_samples = int(additional_params[2])
            changepoint_prior_scale = float(additional_params[3])
        else:
            seasonality_mode = "additive"
            seasonality_prior_scale = 10.0
            uncertainty_samples = 1000
            changepoint_prior_scale = 0.01

        # Detección de outliers
        if detect_outliers:
            outliers = detect_outliers_func(df['y'])
            df['outliers'] = outliers

        # Configurar el modelo Prophet
        model = Prophet(
            weekly_seasonality=freq_is_daily,
            daily_seasonality=True if freq_is_daily else False,  # Estacionalidad semanal solo para datos diarios
            yearly_seasonality=False if freq_is_daily else True,  # Siempre agregar estacionalidad anual
            seasonality_mode=seasonality_mode,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=uncertainty_samples
        )

        # Estacionalidades adicionales
        if freq_is_daily:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        else:
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)

        # Ajustar el modelo, omitiendo outliers si es necesario
        if detect_outliers:
            model.fit(df[~df['outliers']])
        else:
            model.fit(df)

        # Crear fechas futuras con frecuencia adecuada
        future = model.make_future_dataframe(periods=prediction_periods, freq='D' if freq_is_daily else 'MS')
        future['cap'] = max_cap

        # Predecir
        forecast = model.predict(future)

        # Obtener predicciones para entrenamiento
        train_predictions_df = model.predict(df)
        train_predictions = train_predictions_df[['ds', 'yhat']].tail(len(dates))['yhat'].values

        # Predicciones futuras
        future_predictions = forecast['yhat'].tail(prediction_periods).values
        limit = avg_historical * explosive

        # Combinación de predicciones y manejo de valores negativos
        total_predictions = list(train_predictions) + list(future_predictions)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]

        if explosive > 0.0:
            total_predictions = [prediction if prediction <= limit else limit for prediction in total_predictions]

        return idx, total_predictions

    # ASD
    @staticmethod
    def lasso(idx, row, test_periods, prediction_periods, explosive):
        time_series = pd.Series(row).astype(dtype='float')
        avg_historical = time_series.mean()
        train_data = time_series[:-test_periods]
        test_data = time_series.iloc[-test_periods:]

        model = Lasso(alpha=1.0)
        x_train = pd.DataFrame(pd.to_numeric(pd.to_datetime(train_data.index))).astype(int).values.reshape(-1, 1)
        y_train = train_data.values
        model.fit(x_train, y_train)

        x_test = pd.DataFrame(pd.to_numeric(pd.to_datetime(test_data.index))).astype(int).values.reshape(-1, 1)
        test_predictions = model.predict(x_test)
        train_predictions = model.predict(x_train)

        last_date = pd.to_datetime(time_series.index[-1])
        future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, prediction_periods + 1)]
        x_future = pd.DataFrame(pd.to_numeric(pd.to_datetime(future_dates))).astype(int).values.reshape(-1, 1)
        future_predictions = model.predict(x_future)
        total_predictions = list(train_predictions) + list(test_predictions) + list(future_predictions)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
        
        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

        return idx, total_predictions

    def decision_tree(idx, row, test_periods, prediction_periods, dates, is_daily=False, explosive=0):
        try:
            # Crear una serie de tiempo con índices de fecha
            freq = 'D' if is_daily else 'MS'
            dates = pd.date_range(start=dates[0], periods=len(row), freq=freq)
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            avg_historical = time_series.mean()

            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Crear características adicionales
            def create_features(index):
                features = pd.DataFrame(index=index)
                features['day'] = index.day if is_daily else 0  # Solo si es diario
                features['month'] = index.month
                features['year'] = index.year
                features['timestamp'] = pd.to_numeric(index)
                return features

            # Normalizar los datos
            scaler_y = MinMaxScaler()
            y_train = train_data.values.reshape(-1, 1)
            y_train_scaled = scaler_y.fit_transform(y_train)

            x_train = create_features(train_data.index)
            x_test = create_features(test_data.index)

            scaler_x = MinMaxScaler()
            x_train_scaled = scaler_x.fit_transform(x_train)
            x_test_scaled = scaler_x.transform(x_test)

            # Entrenar el modelo
            model = DecisionTreeRegressor(random_state=42, max_depth=5)
            model.fit(x_train_scaled, y_train_scaled)

            # Predicciones para los datos de prueba
            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)))

            # Predicciones para los datos de entrenamiento
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1)))

            # Generar fechas futuras
            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = pd.date_range(start=last_date + pd.DateOffset(days=1 if is_daily else 30), 
                                        periods=prediction_periods, freq=freq)
            x_future = create_features(future_dates)
            x_future_scaled = scaler_x.transform(x_future)

            # Predicciones futuras
            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1)))

            # Evitar predicciones negativas
            future_predictions = [max(pred, 0) for pred in future_predictions]
            total_predictions = list(train_predictions) + list(test_predictions) + list(future_predictions)
            total_predictions = [max(prediction, 0) for prediction in total_predictions]
            
            if explosive > 0.0:
                total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

            return idx, total_predictions

        except Exception as err:
            return str(err)


        except Exception as err:
            return str(err)

    @staticmethod
    def bayesian(idx, row, test_periods, prediction_periods, dates, explosive):
        try:
            # Crear una serie de tiempo con índices de fecha
            dates = pd.date_range(start=dates[0], periods=len(row), freq='MS')
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            avg_historical = time_series.mean()

            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Normalizar las fechas y los datos
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            x_train = pd.to_numeric(pd.to_datetime(train_data.index)).values.reshape(-1, 1)
            y_train = train_data.values.reshape(-1, 1)

            x_train_scaled = scaler_x.fit_transform(x_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

            model = BayesianRidge()
            model.fit(x_train_scaled, y_train_scaled.ravel())

            x_test = pd.to_numeric(pd.to_datetime(test_data.index)).values.reshape(-1, 1)
            x_test_scaled = scaler_x.transform(x_test)

            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled.reshape(-1, 1)))
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1)))

            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_periods + 1)]
            x_future = pd.to_numeric(pd.to_datetime(future_dates)).values.reshape(-1, 1)
            x_future_scaled = scaler_x.transform(x_future)

            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled.reshape(-1, 1)))

            # Evitar predicciones negativas
            future_predictions = [max(pred, 0) for pred in future_predictions]
            total_predictions = list(train_predictions) + list(test_predictions) + list(future_predictions)
            total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
            
            if explosive > 0.0:
                total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]
            return idx, total_predictions

        except Exception as err:
            print(err)

    @staticmethod
    def linear(idx, row, test_periods, prediction_periods, dates, explosive):
        try:
            dates = pd.date_range(start=dates[0], periods=len(row), freq='MS')
            time_series = pd.Series(row, index=dates).astype(dtype='float')
            avg_historical = time_series.mean()

            # Dividir en datos de entrenamiento y prueba
            train_data = time_series[:-test_periods]
            test_data = time_series.iloc[-test_periods:]

            # Normalizar las fechas y los datos
            scaler_x = MinMaxScaler()
            scaler_y = MinMaxScaler()

            x_train = pd.to_numeric(pd.to_datetime(train_data.index)).values.reshape(-1, 1)
            y_train = train_data.values.reshape(-1, 1)

            x_train_scaled = scaler_x.fit_transform(x_train)
            y_train_scaled = scaler_y.fit_transform(y_train)

            model = LinearRegression()
            model.fit(x_train_scaled, y_train_scaled)

            x_test = pd.to_numeric(pd.to_datetime(test_data.index)).values.reshape(-1, 1)
            x_test_scaled = scaler_x.transform(x_test)

            test_predictions_scaled = model.predict(x_test_scaled)
            test_predictions = np.squeeze(scaler_y.inverse_transform(test_predictions_scaled))
            train_predictions_scaled = model.predict(x_train_scaled)
            train_predictions = np.squeeze(scaler_y.inverse_transform(train_predictions_scaled))

            last_date = pd.to_datetime(time_series.index[-1])
            future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, prediction_periods + 1)]
            x_future = pd.to_numeric(pd.to_datetime(future_dates)).values.reshape(-1, 1)
            x_future_scaled = scaler_x.transform(x_future)

            future_predictions_scaled = model.predict(x_future_scaled)
            future_predictions = np.squeeze(scaler_y.inverse_transform(future_predictions_scaled))

            # Evitar predicciones negativas
            # future_predictions = [max(pred, 0) for pred in future_predictions]
            total_predictions = list(train_predictions) + list(test_predictions) + list(future_predictions)
            total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
            
            if explosive > 0.0:
                total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]

            return idx, total_predictions

        except Exception as err:
            print(err)

    @staticmethod
    def neural_prophet(idx, row, prediction_periods, dates, explosive):
        # Asegurarse de que las listas de fechas y valores tengan la misma longitud
        if len(dates) != len(row):
            raise ValueError(f"Las longitudes de las fechas y los valores no coinciden para el índice {idx}")

        # Crear un DataFrame con la fecha y la columna objetivo
        df = pd.DataFrame({'ds': dates, 'y': row})
        avg_historical = df['y'].mean()

        # Verificar si la columna 'y' tiene valores nulos
        if df['y'].isnull().any():
            raise ValueError(f"Hay valores nulos en la columna 'y' para el índice {idx}")

        # Verificar si la columna 'y' tiene valores constantes
        if df['y'].nunique() == 1:
            constant_value = df['y'].iloc[0]
            # Si todos los valores son constantes, devuelve el mismo valor para todas las predicciones
            return idx, [constant_value] * (len(row) + prediction_periods)

        # Inicializar el modelo NeuralProphet con ajustes adicionales
        model = NeuralProphet(
            epochs=75,  # Aumentar el número de épocas para un mejor ajuste
            learning_rate=0.05,  # Ajustar la tasa de aprendizaje
            seasonality_mode='multiplicative',  # Usar la estacionalidad aditiva
            yearly_seasonality=True,  # Habilitar estacionalidad anual
            weekly_seasonality=False,  # Habilitar estacionalidad semanal
            daily_seasonality=False,  # Deshabilitar estacionalidad diaria (si no es relevante)
            n_changepoints=15  # Aumentar el número de puntos de cambio
        )

        # Entrenar el modelo
        model.fit(df, freq='M')

        # Obtener predicciones del conjunto de entrenamiento
        train_predictions = model.predict(df)['yhat1']

        # Hacer predicciones para el período futuro
        future = model.make_future_dataframe(df, periods=prediction_periods)
        future_predictions = model.predict(future)['yhat1'][-prediction_periods:]

        # Combinar predicciones de entrenamiento y futuras
        combined_predictions = list(train_predictions) + list(future_predictions)
        total_predictions = list(train_predictions) + list(future_predictions)
        total_predictions = [prediction if prediction >= 0 else 0 for prediction in total_predictions]
        
        if explosive > 0.0:
            total_predictions = [prediction if prediction <= avg_historical*explosive else avg_historical*explosive for prediction in total_predictions]
        
        return idx, total_predictions