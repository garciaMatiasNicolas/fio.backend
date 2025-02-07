import pandas as pd
import multiprocessing
from .ForecastModels import ForecastModels
from .Error import Error
from file.models import Sales, Product, File, ProjectedExogenousVariables
from joblib import Parallel, delayed
import warnings
from .models import Scenario, PredictedSale, MetricsScenarios
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import date
from django.db.models import Q, F
from clients.models import Clients
from projects.models import Projects
import statistics


warnings.filterwarnings("ignore")


class Forecast(object):

    def __init__(self, scenario: Scenario):

        self.scenario: Scenario = scenario
        self.client: Clients = scenario.client
        self.project: Projects = scenario.project
        self.prediction_periods: int = scenario.pred_p
        self.additional_params: dict = scenario.additional_params
        self.error_periods: int = scenario.error_p
        self.error_method: str = scenario.error_type
        self.seasonal_periods: int = scenario.seasonal_periods
        self.models: list = scenario.models
        self.detect_outliers: bool = scenario.detect_outliers
        self.explosive: float = scenario.explosive
        self.is_daily: bool = scenario.is_daily 
        self.exogenous_values: pd.DataFrame = None
        self.projected_exogenous_values: pd.DataFrame = None
        self.filter_products: bool = self.scenario.filter_products

        if scenario.test_p == 1:
            self.test_periods: int = 2
        else:
            self.test_periods: int = scenario.test_p

        self.initial_dates: list = []
        self.all_dates: list = []
        self.max_date_index: int = None
    
    @staticmethod
    def _get_next_month_date(current_date, months_ahead):
        year = current_date.year + (current_date.month + months_ahead - 1) // 12
        month = (current_date.month + months_ahead - 1) % 12 + 1

        next_month_date = date(year, month, 1)
        return next_month_date

    def get_historical_data(self):
        if any(model in ['prophet_exog', 'arimax', 'sarimax', 'prophet_holidays'] for model in self.models):
            sales_data = Sales.objects.filter(
                product__file__project=self.project,
                product__file__project__client=self.client,
                product__discontinued=False,
                product__avg__gt=0 if self.filter_products else -1
            ).annotate(
                exog_value=F('sale_related__exog'),
                variable=F('sale_related__variable')
            ).values(
                'product_id', 'date', 'sale', 'exog_value', 'variable'
            ).order_by('product_id', 'date')

            df = pd.DataFrame(data=list(sales_data))

            df_exog = df.copy()
            df_exog['exog_value'] = df_exog['exog_value'].fillna(0.0)
            df_exog.dropna(subset=['variable'], inplace=True)
            df_exog.drop(columns=['sale'], inplace=True)

            self.exogenous_values = df_exog
        
            df.drop(columns=['exog_value', 'variable'], inplace=True)
            df.drop_duplicates(inplace=True)

            projected = File.objects.filter(project=self.project, project__client=self.client, file_type='projected_exogenous').first()

            if projected is not None:
                projected = ProjectedExogenousVariables.objects.filter(
                    product__file__project=self.project,
                    product__file__project__client=self.client,
                    product__discontinued=False,
                    product__avg__gt=0 if self.filter_products else -1
                ).values(
                    'product_id', 'date', 'exog', 'variable'
                ).order_by('product_id', 'date')

                df_projected = pd.DataFrame(data=list(projected))
                df_projected.dropna(subset=['variable'], inplace=True)

                self.projected_exogenous_values = df_projected

        else:
            sales_data = Sales.objects.filter(
                product__file__project=self.project,
                product__file__project__client=self.client,
                product__discontinued=False,
                product__avg__gt=0 if self.filter_products else -1
            ).values(
                'product_id', 'date', 'sale'
            ).order_by('product_id', 'date')

            df = pd.DataFrame(data=list(sales_data))

        # Get dates and drop column
        self.initial_dates = df['date'].drop_duplicates().tolist()
        self.max_date_index = len(self.initial_dates)-1
        last_initial_date = self.initial_dates[-1]
        self.all_dates = self.initial_dates.copy()
        
        # Add the predicted dates
        for i in range(self.prediction_periods):
            next_month_date = self._get_next_month_date(last_initial_date, i+1)
            self.all_dates.append(next_month_date)
        
        df = df.drop(columns=['date'])
        sales_dict = df.groupby('product_id')['sale'].apply(list).to_dict()
        sales_dict = {product_id: sales for product_id, sales in sales_dict.items()}

        return sales_dict

    def parallel_process(self, data, model_name: str, actual: dict, exog_data=None):
        num_cores = int(multiprocessing.cpu_count() * 0.8)
        data_list = [(product_id, sales) for product_id, sales in data.items()]

        if model_name in ["holtsWintersExponentialSmoothing", "holtsExponentialSmoothing", "exponential_moving_average"]:
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.holt_holtwinters_ema)(
                    prod_id, 
                    row, 
                    self.test_periods, 
                    self.prediction_periods,
                    model_name, 
                    self.seasonal_periods, 
                    is_daily=self.is_daily, 
                    explosive=self.explosive
                ) for prod_id, row in data_list
            )

        elif model_name == "arimax":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.arima)(
                    idx=prod_id, 
                    row=row, 
                    test_periods=self.test_periods, 
                    prediction_periods=self.prediction_periods, 
                    additional_params=self.additional_params, 
                    explosive=self.explosive
                ) for prod_id, row in data_list
            )
        
        elif model_name == "arima":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.arima)(
                    idx=prod_id, 
                    row=row, 
                    test_periods=self.test_periods, 
                    prediction_periods=self.prediction_periods, 
                    additional_params=self.additional_params, 
                    explosive=self.explosive
                ) for prod_id, row in data_list
            )

        elif model_name in ["AutoARIMA", "Chronos", "NPTS"]:
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.autogluon)(
                    idx=prod_id, 
                    row=row, 
                    prediction_periods=self.prediction_periods, 
                    dates=self.initial_dates,
                    explosive=self.explosive,
                    model=model_name
                ) for prod_id, row in data_list
            )
        
        elif model_name == "sarimax":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.sarimax)(idx=prod_id, row=row, exog_data=exog_data, test_periods=self.test_periods, prediction_periods=self.prediction_periods,additional_params=self.additional_params, explosive=self.explosive)
                for prod_id, row in data_list)
        
        elif model_name == "prophet_holidays":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.prophet_holidays)(
                    idx=prod_id, 
                    row=row, 
                    exog_data=self.exogenous_values.loc[self.exogenous_values['product_id'] == prod_id], 
                    prediction_periods=self.prediction_periods, 
                    dates=self.initial_dates, 
                    explosive=self.explosive, 
                    monthly=not self.is_daily,
                    detect_outliers=self.detect_outliers
                )for prod_id, row in data_list) 
        
        elif model_name == "prophet_exog":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.prophet_exog)(
                    idx=prod_id, 
                    row=row, 
                    exog_data=self.exogenous_values.loc[self.exogenous_values['product_id'] == prod_id], 
                    prediction_periods=self.prediction_periods, 
                    dates=self.initial_dates, 
                    explosive=self.explosive, 
                    projected_exog_data=self.projected_exogenous_values.loc[self.projected_exogenous_values['product_id'] == prod_id] if self.projected_exogenous_values is not None else None,
                    monthly=not self.is_daily
                )for prod_id, row in data_list) 

        elif model_name == "sarima":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.sarima)(
                    idx=prod_id, 
                    row=row, 
                    test_periods=self.test_periods,
                    prediction_periods=self.prediction_periods,
                    additional_params=self.additional_params, 
                    explosive=self.explosive
                ) 
            for prod_id, row in data_list
        )

        elif model_name == "simpleExponentialSmoothing":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.exponential_smoothing)(prod_id, row, self.test_periods, self.prediction_periods, explosive=self.explosive)
                for prod_id, row in data_list)

        elif model_name == "prophet":
            additional_params = self.additional_params.get("prophet_params", None)
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.prophet)(
                    idx=prod_id, 
                    row=row, 
                    prediction_periods=self.prediction_periods, 
                    additional_params=additional_params, 
                    seasonal_periods=self.seasonal_periods, 
                    dates=self.initial_dates, 
                    detect_outliers=self.detect_outliers, 
                    explosive=self.explosive
                )
                for prod_id, row in data_list)
        
        elif model_name == "neuralProphet":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.neural_prophet)(prod_id, row, self.prediction_periods, self.initial_dates, explosive=self.explosive)
                for prod_id, row in data_list)

        elif model_name == "decisionTree":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.decision_tree)(prod_id, row, self.test_periods, self.prediction_periods, self.initial_dates, is_daily=self.is_daily, explosive=self.explosive)
                for prod_id, row in data_list)

        elif model_name == "lasso":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.lasso)(prod_id, row, self.test_periods, self.prediction_periods, explosive=self.explosive)
                for prod_id, row in data_list)

        elif model_name == "linearRegression":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.linear)(prod_id, row, self.test_periods, self.prediction_periods, self.initial_dates, explosive=self.explosive)
                for prod_id, row in data_list)

        elif model_name == "bayesian":
            results = Parallel(n_jobs=round(num_cores))(
                delayed(ForecastModels.bayesian)(prod_id, row, self.test_periods, self.prediction_periods, self.initial_dates, explosive=self.explosive)
                for prod_id, row in data_list)

        predicted = {}
        metrics = []
        kpis_data = {}

        for product_id, result in results:
            actual_sales = actual[product_id]
            predicted_sales = [round(num, 2) for num in result[:len(actual_sales)]]
            combined_sales = actual_sales + result[len(actual_sales):]

            error = Error(error_method=self.error_method, error_periods=self.error_periods, actual=actual_sales,
                        predicted=predicted_sales)

            error, last_period_error = error.calculate_error_periods_selected()
            ## Metrics Object ##
            metrics.append({ 
                "scenario": self.scenario,
                "product_id": product_id,
                "error": error, 
                "last_error": last_period_error,
                "model": model_name, 
            })

            predicted[product_id] = result
            kpis_data[product_id] = combined_sales
            

        return predicted, metrics, kpis_data

    @staticmethod
    def select_best_model(df: pd.DataFrame):
        df = df.copy()
        df['best_model'] = False

        ## Choose best model by lowest error
        idx_min_error = df.groupby('product_id')['error'].idxmin()
        df.loc[idx_min_error, 'best_model'] = True
        
        return df

    def calculate_kpis(self, data: dict):
        results = {} 

        def calc_perc(n1, n2):
            try:
                if n2 == 0 or n1 is None or n2 is None:  
                    return 0  
                result = round((n1 - n2) / n2 * 100)
                return result
            except (ZeroDivisionError, ValueError):
                return 0  

        for model_name, products in data.items():
            model_results = {}
            
            for product_id, values in products.items():
                # Determinar el tamaño de los valores disponibles
                available_values = len(values)
                
                # Si no hay suficientes datos, ajustamos las ventanas de cálculo
                if available_values >= 24:
                    actual_year_start = self.max_date_index - 11  
                    actual_year_end = self.max_date_index + 1     
                    
                    # Predicted year
                    predicted_year_start = self.max_date_index + 1 
                    predicted_year_end = self.max_date_index + 13   
                    
                    # Quarterly actual
                    q_actual_start = actual_year_start
                    q_actual_end = q_actual_start + 4  
                    
                    # Quarterly predicted
                    q_predicted_start = predicted_year_start
                    q_predicted_end = q_predicted_start + 4  # 
                    
                    # Monthly values
                    m_actual = values[self.max_date_index]  
                    m_predicted = values[self.max_date_index + 1]  
                
                # Sumas
                actual_year = sum(values[actual_year_start:actual_year_end])
                predicted_year = sum(values[predicted_year_start:predicted_year_end])
                q_actual = sum(values[q_actual_start:q_actual_end])
                q_predicted = sum(values[q_predicted_start:q_predicted_end])

                # Cálculos de KPIs
                ytg = calc_perc(n1=predicted_year, n2=actual_year)
                qtg = calc_perc(n1=q_predicted, n2=q_actual)
                mtg = calc_perc(n1=m_predicted, n2=m_actual)

                if actual_year_end > actual_year_start:  # Asegurarse de que el rango sea válido
                    subset_values = values[actual_year_start:actual_year_end]
                    avg_actual_year = sum(subset_values) / len(subset_values)
                    std_dev_actual_year = statistics.stdev(subset_values)
                else:
                    avg_actual_year = 0 

                model_results[product_id] = {
                    "ytg": ytg,
                    "qtg": qtg,
                    "mtg": mtg, 
                    "avg": avg_actual_year,
                    "desv": std_dev_actual_year
                }
            
            results[model_name] = model_results
        
        return results
    
    @staticmethod
    def cluster_products(products: pd.DataFrame):

        scaler = StandardScaler()

        try:
            # Escalado de valores
            scaled_values = scaler.fit_transform(products[['error', 'avg']])
            products[['error_Scaled', 'avg_Scaled']] = scaled_values
            products['sales_ratio'] = products['avg'] / products['desv']

            unique_data_points = len(products[['error_Scaled', 'avg_Scaled']].drop_duplicates())
            n_clusters = min(3, unique_data_points)

            # Aplicar KMeans
            kmeans_mape = KMeans(n_clusters=n_clusters, random_state=42)
            products['Cluster'] = kmeans_mape.fit_predict(products[['error_Scaled', 'avg_Scaled']])

            # Mapear etiquetas de riesgo
            risk_labels = {
                i: label for i, label in enumerate(['Bajo riesgo', 'Mediano riesgo', 'Alto riesgo'][:n_clusters])
            }
            products['Cluster'] = products['Cluster'].map(risk_labels)

            # Calcular clasificación ABC
            products = products.sort_values(by='avg', ascending=False).reset_index(drop=True)
            total_sales = products['avg'].sum()
            cumulative_sales = products['avg'].cumsum() / total_sales

            products['ABC'] = 'C'
            products.loc[cumulative_sales <= 0.8, 'ABC'] = 'A'
            products.loc[(cumulative_sales > 0.8) & (cumulative_sales <= 0.9), 'ABC'] = 'B'

            # Limpiar columnas temporales
            products = products.drop(columns=['avg_Scaled', 'sales_ratio', 'error_Scaled'])

            return products.to_dict(orient='records')

        except ValueError as e:
            raise ValueError("Error en el cálculo de clusters: " + str(e))

    def run_forecast(self):
        initial_data = self.get_historical_data()
        all_models_metrics = []
        all_kpi_results = {}

        for model_name in self.models:
            forecast_results, metrics, kpis = self.parallel_process(data=initial_data, model_name=model_name, actual=initial_data)
            objects = []

            all_models_metrics.append(metrics)
            all_kpi_results[model_name] = kpis

            for product_id, sales in forecast_results.items():
                product = Product.objects.get(id=product_id)  
                for sale, date in zip(sales, self.all_dates):
                    objects.append(
                        PredictedSale (
                            scenario=self.scenario, 
                            product=product,
                            sale=sale,
                            model=model_name,  
                            date=date,
                            colaborated_sale=sale
                        )
                    )
            
            PredictedSale.objects.bulk_create(objects, batch_size=10000)
        
        ## Prepare data for best_model select
        all_models_metrics = [item for sublist in all_models_metrics for item in sublist]
        df = pd.DataFrame(all_models_metrics) 

        ## Select best model
        metrics_data = self.select_best_model(df=df)

        ## Calculate kpi for best model
        kpis_data = self.calculate_kpis(data=all_kpi_results)
        
        ## Concat results to metrics
        def get_metrics(row):
            model = row['model']
            product_id = row['product_id']
            return kpis_data.get(model, {}).get(product_id, {'ytg': None, 'qtg': None, 'mtg': None, 'avg': None, 'desv': None})
        
        metrics_df = metrics_data.apply(get_metrics, axis=1, result_type='expand')
        metrics_data = pd.concat([metrics_data, metrics_df], axis=1)
        metrics_data = self.cluster_products(products=metrics_data)

        metrics = []
        conditions = Q()

        for data in metrics_data:
            product = Product.objects.get(pk=data["product_id"])
            
            if data['best_model'] == True:
                conditions |= Q(product_id=data['product_id'], model=data['model'])
            
            ytg = data['ytg'] if data['ytg'] is not None and -2147483648 <= data['ytg'] <= 2147483647 else 0
            qtg = data['qtg'] if data['qtg'] is not None and -2147483648 <= data['qtg'] <= 2147483647 else 0
            mtg = data['mtg'] if data['mtg'] is not None and -2147483648 <= data['mtg'] <= 2147483647 else 0


            metrics.append(
                MetricsScenarios(
                    scenario=data['scenario'],
                    product=product,
                    best_model=data['best_model'],
                    last_period_error=data['last_error'],
                    model=data['model'],
                    error=data['error'],
                    ytg=ytg,
                    qtg=qtg,
                    mtg=mtg,
                    cluster=data['Cluster'],
                    abc=data['ABC']
                )
            )

        sales_to_update = PredictedSale.objects.filter(conditions)
        
        for sale in sales_to_update:
            sale.best_model = True

        PredictedSale.objects.bulk_update(sales_to_update, ['best_model'], batch_size=10000)
        MetricsScenarios.objects.bulk_create(metrics, batch_size=10000)


        


