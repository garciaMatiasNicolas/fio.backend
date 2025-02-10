from rest_framework import status
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from .serializer import ForecastScenarioSerializer
from .models import Scenario, PredictedSale, MetricsScenarios
from projects.models import Projects
from rest_framework.views import APIView
from .Forecast import Forecast
from rest_framework.permissions import IsAuthenticated
from django.db.models import Sum, Avg, Sum, Func, Q, F, OuterRef, Subquery, Max, DecimalField, ExpressionWrapper
from django.db.models.functions import ExtractYear, TruncDate, Coalesce, ExtractMonth
from datetime import date
from file.models import Product, Sales, File
from django.core.paginator import Paginator
from rest_framework.response import Response
import traceback
import pandas as pd
from django.shortcuts import get_object_or_404
from .Error import Error
from collections import Counter
import datetime
from dateutil.relativedelta import relativedelta
from django.utils.timezone import now
from datetime import timedelta
from django.core.exceptions import ObjectDoesNotExist


class ForecastScenarioViewSet(ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = ForecastScenarioSerializer
    queryset = Scenario.objects.all()

    def list(self, request, *args, **kwargs):
        client = request.user.userinformation.get().client
        queryset = self.get_queryset().filter(project__client=client, project__name=request.GET.get('project')).order_by('-created_at')
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            scenario = serializer.save()
            forecast = Forecast(scenario=scenario)

            try:
                forecast.run_forecast()

            except Exception as err:
                traceback.print_exc()
                scenario.delete()
                return Response({"error": str(err)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response(
                {'message': 'scenario_runned_successfully', 'scenario_id': scenario.id},
                status=status.HTTP_201_CREATED
            )
        
        else:
            return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def destroy(self, request, *args, **kwargs):
        return super().destroy(request, *args, **kwargs)


class ColaborationViews:

    class GetPredictedDatesAPIView(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            client = request.user.userinformation.get().client

            try:
                if scenario_name is not None:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()

                if project is not None:
                    max_historical_date = Projects.objects.filter(client=client, name=project).aggregate(
                        max_date=Coalesce(Max('max_historical_date'), date.min)
                    )['max_date']
                else:
                    project = Scenario.objects.filter(name=scenario_name).values_list('project__name', flat=True).first()
                    max_historical_date = Projects.objects.filter(client=client, name=project).aggregate(
                        max_date=Coalesce(Max('max_historical_date'), date.min)
                    )['max_date']

            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

            if max_historical_date:
                dates = PredictedSale.objects.filter(
                    scenario=scenario,
                    date__gt=max_historical_date
                ).values_list('date', flat=True).distinct()
            else:
                dates = PredictedSale.objects.filter(
                    scenario=scenario
                ).values_list('date', flat=True).distinct()

            return Response(dates, status=status.HTTP_200_OK)


    class GetColaborationChartAPIView(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            group = request.GET.get('group')
            value = request.GET.get('value')
            client = request.user.userinformation.get().client
 
            try:
                if scenario_name is not None:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)
            
            products = Product.objects.filter(
                avg__gte=0 if scenario.filter_products else -1,
                file__project__client=client,
                file__project__name=project, 
                **{f"{group}__icontains": value}
            ).values_list('id', flat=True)

            predicted_sales =  PredictedSale.objects.filter(product__id__in=products, scenario=scenario, best_model=True)
            actual_sales = Sales.objects.filter(product__id__in=products)

            actual_data = (
                actual_sales
                .annotate(truncated_date=TruncDate('date'))
                .values('truncated_date')
                .annotate(total_sales=Func(Sum('sale'), function='ROUND', template='%(function)s(%(expressions)s, 2)'))
                .order_by('truncated_date')
            )

            predicted_data = (
                predicted_sales
                .annotate(truncated_date=TruncDate('date'))
                .values('truncated_date')
                .annotate(
                    total_sales=Func(Sum('sale'), function='ROUND', template='%(function)s(%(expressions)s, 2)'),
                    colaborated=Func(Sum('colaborated_sale'), function='ROUND', template='%(function)s(%(expressions)s, 2)')
                )
                .order_by('truncated_date')
            )

            data = {
                'predicted': predicted_data,
                'actual': actual_data
            }

            return Response(data=data, status=status.HTTP_200_OK)


    class ListColaborationDataAPIView(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            date = request.GET.get('date')
            group = request.GET.get('group')
            value = request.GET.get('value')
            client = request.user.userinformation.get().client 

            try:
                if scenario_name is not None:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)
            
            products = Product.objects.filter(
                avg__gte=0 if scenario.filter_products else -1,
                file__project__client=client, 
                file__project__name=project, 
                **{group: value}
            ).values_list('id', flat=True)
            
            try:
                current_date = datetime.datetime.strptime(date, "%Y-%m-%d")  
                last_month_date = current_date - relativedelta(months=1)

                same_date_last_year = current_date - relativedelta(years=1)  
            except ValueError:
                return Response({"error": "Invalid date format."}, status=status.HTTP_400_BAD_REQUEST)

            same_date_all_years_sales = Sales.objects.filter(
                product__id__in=products,
                date__month=current_date.month,
                date__day=current_date.day
            ).annotate(year=ExtractYear('date')).values('year').annotate(total_sales=Sum('sale')).order_by('-year')

            # Filtrar las ventas para la misma fecha del mes pasado
            sales_data = Sales.objects.filter(
                product__id__in=products, 
                date=last_month_date.date() 
            )

            last_year_date = PredictedSale.objects.filter(
                product__id__in=products, 
                date=same_date_last_year.date() 
            )

            predicted_data = PredictedSale.objects.filter(
                scenario=scenario,
                product__id__in=products, 
                date=date,
                best_model=True
            )

            last_month = sales_data.aggregate(total_sales=Sum('sale'))['total_sales'] or 0
            predicted = predicted_data.aggregate(total=Sum('sale'))['total'] or 0
            colaborated = predicted_data.aggregate(total=Sum('colaborated_sale'))['total'] or 0
            same_date_last_year = last_year_date.aggregate(total=Sum('sale'))['total'] or 0
            updated_at = predicted_data.aggregate(updated_date=Max('updated_at'))['updated_date']

            return Response({
                "previous_years": list(same_date_all_years_sales),
                "same_date_last_year": round(same_date_last_year, 2),
                "last_month": round(last_month, 2),
                "predicted": round(predicted, 2),
                "colaborated": round(colaborated, 2),
                "updated_at": updated_at
            })
        

    class ColaborationForecastAPIView(APIView):
        permission_classes = [IsAuthenticated]

        @staticmethod
        def rate(pred: float, total: float, new_sale: float):
            return (pred / total) * new_sale if total != 0 else 0.0

        @staticmethod
        def percentage(pred: float, new_sale: float):
            return pred + (pred * new_sale / 100)

        def put(self, request):
            data = request.data
            action = data.get("action")
            date = data.get("date")
            new_sale = float(data.get("new_sale"))
            value = data.get("value")
            group = data.get("group")
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            client = request.user.userinformation.get().client 

            try:
                if scenario_name is not None:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

            products = Product.objects.filter(
                avg__gte=0 if scenario.filter_products else -1,
                file__project__client=client, 
                file__project__name=project, 
                **{group: value}
            ).values_list('id', flat=True)

            sales_with_best_model = PredictedSale.objects.filter(
                product_id__in=products,
                scenario=scenario,
                best_model=True,
                date=date,
            ).values('id', 'product_id', 'sale', 'updated_at', 'colaborated_sale', 'updated_by')

            total = sum(product['sale'] for product in sales_with_best_model) or 0.0

            if total == 0:
                return Response({"message": "No sales data to process."}, status=status.HTTP_200_OK)

            # Prepare objects for bulk_update
            updated_objects = []
            for product in sales_with_best_model:
                pred = PredictedSale.objects.get(id=product['id'])
                if product['updated_at'] is None:
                    if action == "rate":
                        colaborated_sale = self.rate(pred=product['sale'], total=total, new_sale=new_sale)
                    elif action == "percentage":
                        colaborated_sale = self.percentage(pred=product['sale'], new_sale=new_sale)
                else:
                    if action == "rate":
                        colaborated_sale = self.rate(pred=product['colaborated_sale'], total=total, new_sale=new_sale)
                    elif action == "percentage":
                        colaborated_sale = self.percentage(pred=product['colaborated_sale'], new_sale=new_sale)
                
                pred.colaborated_sale = colaborated_sale
                pred.updated_at = now()
                pred.updated_by = request.user
                updated_objects.append(pred)

            # Perform bulk_update
            PredictedSale.objects.bulk_update(updated_objects, ['colaborated_sale', 'updated_at', 'updated_by'])

            return Response({"message": "Escenario colaborado exitosamente"}, status=status.HTTP_200_OK)
        

class AnalyticsViewsForScenarios:
    

    class ForecastAnalyticsAPIView(APIView):
        permission_classes = [IsAuthenticated]

        @staticmethod
        def apply_product_filters(queryset, request):
            '''
            Aplica los filtros de producto al queryset según los parámetros de la solicitud,
            asegurando que funcionen correctamente de manera anidada.
            '''
            
            filter_fields = [
                'family', 'salesman', 'region', 'client',
                'category', 'subcategory', 'sku'
            ]

            filters = {
                f"{field}__in": request.GET.getlist(field)
                for field in filter_fields
                if request.GET.getlist(field)
            }

            return queryset.filter(**filters)

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            type_of = request.GET.get('typeofget')
            conversion = request.GET.get('conversion', None)
            project = request.GET.get('project')
            client = request.user.userinformation.get().client 

            max_historical_date = (
                Projects.objects.filter(client=client, name=project)
                .aggregate(max_date=Coalesce(Max('max_historical_date'), date.min))['max_date']
            )

            if scenario_name is not None:
                scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
            else:
                scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()

            if scenario is None:
                return Response({'message': 'No se encontraron escenarios para el proyecto'}, status=status.HTTP_400_BAD_REQUEST)

            products = self.apply_product_filters(
                Product.objects.filter(avg__gte=0 if scenario.filter_products else -1, file__project__client=client, file__project__name=project).only('id'), request
            )

            stock_data = File.objects.filter(project__client=client, project__name=project, file_type='stock')

            if type_of == 'linegraph':
                warning = None

                if conversion == 'price' and stock_data.exists():
                    actual_sales = Sales.objects.filter(product__in=products).annotate(
                        total_revenue=ExpressionWrapper(
                            F('sale') * F('product__stock__cost_price'),
                            output_field=DecimalField()
                        )
                    )

                    actual_data = list(
                        actual_sales
                        .annotate(truncated_date=TruncDate('date'))
                        .values('truncated_date')
                        .annotate(total_sales=Sum('total_revenue'))
                        .order_by('-truncated_date')
                    )
                    
                    predicted_sales = PredictedSale.objects.filter(
                        scenario=scenario, product__in=products, best_model=True
                    ).annotate(
                        total_revenue=ExpressionWrapper(
                            F('sale') * F('product__stock__cost_price'),
                            output_field=DecimalField()
                        )
                    )

                    if predicted_sales.exists():
                        predicted_data = list(
                            predicted_sales
                            .annotate(truncated_date=TruncDate('date'))
                            .values('truncated_date')
                            .annotate(total_sales=Sum('total_revenue'))
                            .order_by('-truncated_date')
                        )
                    else:
                        predicted_data = [] 
                
                else:
                    actual_sales = Sales.objects.filter(product__in=products)

                    predicted_sales = PredictedSale.objects.filter(
                        scenario=scenario, product__in=products, best_model=True
                    ).only('date', 'sale')
                
                    actual_data = list(
                        actual_sales
                        .annotate(truncated_date=TruncDate('date'))
                        .values('truncated_date')
                        .annotate(total_sales=Sum('sale'))
                        .order_by('-truncated_date')
                    )
                    
                    if predicted_sales.exists():
                        predicted_data = list(
                            predicted_sales
                            .annotate(truncated_date=TruncDate('date'))
                            .values('truncated_date')
                            .annotate(total_sales=Sum('sale'))
                            .order_by('-truncated_date')
                        )
                    else:
                        predicted_data = [] 

                if conversion == 'price' and stock_data.exists() == False:
                    warning = 'No se encontraron datos de stock para calcular el forecast valorizado'
                
                # Estructura de datos de respuesta
                response_data = {
                    'predicted': predicted_data,
                    'actual': actual_data,
                    'warning': warning
                }

            elif type_of == 'bargraph':

                if conversion == 'price' and stock_data.exists():
                    actual_sales = Sales.objects.filter(product__in=products).annotate(
                        total_revenue=ExpressionWrapper(
                            F('sale') * F('product__stock__cost_price'),
                            output_field=DecimalField()
                        )
                    )

                    actual_data = list(
                        actual_sales
                        .annotate(year=ExtractYear('date'))
                        .values('year')
                        .annotate(total_sales=Sum('total_revenue'))
                        .order_by('year')
                    )

                    predicted_sales = PredictedSale.objects.filter(
                        scenario=scenario, 
                        product__in=products, 
                        best_model=True
                    ).annotate(
                        total_revenue=ExpressionWrapper(
                            F('sale') * F('product__stock__cost_price'),
                            output_field=DecimalField()
                        )
                    )

                    if predicted_sales.exists():
                        predicted_data = list(
                            predicted_sales.filter(date__gt=max_historical_date)
                            .annotate(year=ExtractYear('date'))
                            .values('year')
                            .annotate(total_sales=Sum('total_revenue'))
                            .order_by('year')
                        )
                    else:
                        predicted_data = []
                
                else:
                    actual_sales = Sales.objects.filter(product__in=products).only('date', 'sale', 'product_id')
                    actual_data = list(
                        actual_sales
                        .annotate(year=ExtractYear('date'))
                        .values('year')
                        .annotate(total_sales=Sum('sale'))
                        .order_by('year')
                    ) 

                    predicted_sales = PredictedSale.objects.filter(
                        scenario=scenario, 
                        product__in=products, 
                        best_model=True
                    ).only('date', 'sale')

                    if predicted_sales.exists():
                        predicted_data = (
                            list(
                                predicted_sales.filter(date__gt=max_historical_date)
                                .annotate(year=ExtractYear('date'))
                                .values('year')
                                .annotate(total_sales=Sum('sale'))
                                .order_by('year')
                            )
                        )
                    
                    else:
                        predicted_data = [] 

                response_data = {
                    'predicted': predicted_data,
                    'actual': actual_data
                }

            elif type_of == 'error':
                error_data = {}
                error_data = MetricsScenarios.objects.filter(
                    scenario=scenario, best_model=True, product__in=products
                ).aggregate(avg_error=Avg('error'), avg_last_period_error=Avg('last_period_error'))

                response_data = {
                    'error': error_data
                }

            return Response(response_data, status=status.HTTP_200_OK) 


    class ClusterDataTable(APIView):
        permission_classes = [IsAuthenticated]
        
        @staticmethod
        def apply_product_filters(queryset, request):
            """
            Aplica los filtros de producto al queryset según los parámetros de la solicitud,
            asegurando que funcionen correctamente de manera anidada.
            """
            filters = Q()

            # Filtros directos de Product
            product_filter_fields = [
                ('family', 'family__in'),
                ('salesman', 'salesman__in'),
                ('region', 'region__in'),
                ('client', 'client__in'),
                ('category', 'category__in'),
                ('subcategory', 'subcategory__in'),
                ('sku', 'sku__in'),
            ]

            # Aplicar filtros sobre campos de Product
            for field, query in product_filter_fields:
                values = request.GET.getlist(field)
                if values:
                    filters &= Q(**{query: values})

            related_filter_fields = [
                ('cluster', 'metricsscenarios__cluster__in'),
                ('abc', 'metricsscenarios__abc__in'),
            ]

            for field, query in related_filter_fields:
                values = request.GET.getlist(field)
                if values:
                    filters &= Q(**{query: values})

            return queryset.filter(filters).distinct()
        
        def get(self, request):
            export = request.GET.get('export', None)
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            conversion = request.GET.get('conversion', None)
            page_number = int(request.GET.get('page', 1))
            order = request.GET.get('order_by')
            order_type = False if request.GET.get('order_type') == 'desc' else True
            client = request.user.userinformation.get().client 

            try:
                if scenario_name:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

            if scenario.filter_products:
                products = Product.objects.filter(avg__gt=0, file__project__client=client, file__project__name=project)
            
            else:
                products = Product.objects.filter(file__project__client=client, file__project__name=project)

            if export is None:
                products = self.apply_product_filters(products, request)

                if order is not None:
                    order_query = f"-{order}" if order_type == False else order
                    products = products.annotate(
                        AVG=F('avg'),
                        YTD=F('ytd'), 
                        QTD=F('qtd'),  
                        MTD=F('mtd'),
                        YTG=Subquery(
                            MetricsScenarios.objects.filter(
                                product=OuterRef('id'),
                                scenario=scenario,  
                                best_model=True 
                            ).values('ytg')[:1]  
                        ),
                        QTG=Subquery(
                            MetricsScenarios.objects.filter(
                                product=OuterRef('id'),
                                scenario=scenario,  
                                best_model=True  
                            ).values('qtg')[:1]  
                        ),
                        MTG=Subquery(
                            MetricsScenarios.objects.filter(
                                product=OuterRef('id'),
                                scenario=scenario,  
                                best_model=True  
                            ).values('mtg')[:1]  
                        ),
                        Error=Subquery(
                            MetricsScenarios.objects.filter(
                                product=OuterRef('id'),
                                scenario=scenario,  
                                best_model=True  
                            ).values('error')[:1]   
                        )
                    ).order_by(order_query)

                # Paginación
                paginator = Paginator(products, 10)
                page = paginator.get_page(page_number)

                product_ids = list(page.object_list.values_list('id', flat=True))
            
            else:
                product_ids = list(products.values_list('id', flat=True))
            
            products_data = Product.objects.filter(id__in=product_ids).values()
            products_df = pd.DataFrame(products_data)

            if len(products_data) == 0:
                return Response({'message': 'No se encontraron productos para los filtros seleccionados'}, status=status.HTTP_400_BAD_REQUEST)

            # Obtener en un dataframe las fechas de venta historicas
            stock_data = File.objects.filter(project__client=client, project__name=project, file_type='stock')
            
            if conversion == 'price' and stock_data.exists():
                sales = Sales.objects.filter(
                    product_id__in=product_ids
                ).annotate(
                    total_revenue=ExpressionWrapper(
                        F('sale') * F('product__stock__cost_price'),
                        output_field=DecimalField()
                    )
                ).values('product_id', 'date', 'total_revenue')

            else:
                sales = Sales.objects.filter(product_id__in=product_ids).values('product_id', 'date', 'sale')

            sales_df_dates = pd.DataFrame(sales)
            
            if conversion == 'price' and stock_data.exists():
                sales_df_dates.rename(columns={'total_revenue': 'sale'}, inplace=True)

            sales_df_dates['date'] = pd.to_datetime(sales_df_dates['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            pivoted_sales_df = sales_df_dates.pivot(index='product_id', columns='date', values='sale').fillna(0)
            sales_df = pivoted_sales_df.reset_index()

            # Obtener en un dataframe las fechas de venta predecidas
            metrics = MetricsScenarios.objects.filter(
                scenario=scenario,
                product__in=product_ids,
                best_model=True
            ).values('product_id', 'model', 'ytg', 'qtg', 'mtg', 'error', 'cluster', 'abc')

            # Filtrar las ventas predichas e incluir las columnas de best_models
            if conversion == 'price' and stock_data.exists():
                predicted_sales = PredictedSale.objects.filter(
                    scenario=scenario,
                    product__in=product_ids,
                    best_model=True
                ).annotate(
                    ytg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('ytg')[:1]),
                    qtg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('qtg')[:1]),
                    mtg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('mtg')[:1]),
                    cluster=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('cluster')[:1]),
                    error=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('error')[:1]),
                    abc=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('abc')[:1]),
                    total_sale=ExpressionWrapper(
                        F('sale') * F('product__stock__cost_price'),
                        output_field=DecimalField()
                    )
                ).values('product_id', 'date', 'total_sale', 'model', 'ytg', 'qtg', 'mtg', 'error', 'cluster', 'abc')

                print(predicted_sales)

            else:
                predicted_sales = PredictedSale.objects.filter(
                    scenario=scenario,
                    product__in=product_ids,
                    best_model=True
                ).annotate(
                    ytg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('ytg')[:1]),
                    qtg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('qtg')[:1]),
                    mtg=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('mtg')[:1]),
                    cluster=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('cluster')[:1]),
                    error=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('error')[:1]),
                    abc=Subquery(metrics.filter(product_id=OuterRef('product_id')).values('abc')[:1]),
                    total_sale=Func('sale', function='ROUND', template='%(function)s(%(expressions)s, 2)')
                ).values('product_id', 'date', 'total_sale', 'model', 'ytg', 'qtg', 'mtg', 'error', 'cluster', 'abc')

            if len(predicted_sales) == 0:
                return Response({'message': 'No se corrieron escenarios para este proyecto'}, status=status.HTTP_400_BAD_REQUEST)

            predicted_sales_df = pd.DataFrame(predicted_sales)
            predicted_sales_df['date'] = pd.to_datetime(predicted_sales_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

            ## Merge Dataframes ##
            products_df = products_df.drop(columns=['file_id', 'template_id'])
            products_df = products_df.rename(columns={'id': 'product_id'})
            merged_df = pd.merge(products_df, sales_df, on='product_id', how='inner')

            predicted_pivot = predicted_sales_df.pivot_table(
                index=['product_id', 'ytg', 'qtg', 'mtg', 'error', 'cluster', 'abc', 'model'],
                columns='date',
                values='total_sale'
            ).reset_index()
            
            date_cols_df1 = [col for col in merged_df.columns if col.startswith("20")]
            date_cols_df2 = [col for col in predicted_pivot.columns if col.startswith("20")]

            # Encontrar las fechas únicas en df2 que no están en df1
            additional_dates = [col for col in date_cols_df2 if col not in date_cols_df1]

            # Merge por product_id, manteniendo todas las filas de df1
            merged_df = pd.merge(merged_df, predicted_pivot[['product_id', 'ytg', 'qtg', 'mtg', 'error', 'cluster', 'abc', 'model'] + additional_dates], on="product_id", how="left")

            # Ordenar columnas: mantener no-fechas primero, luego todas las fechas ordenadas
            non_date_cols = [col for col in merged_df.columns if not col.startswith("20")]
            all_dates_sorted = sorted([col for col in merged_df.columns if col.startswith("20")])
            merged_df = merged_df[non_date_cols + all_dates_sorted]

            merged_df = merged_df.rename(columns={
                "family": "Familia",
                "region": "Region",
                "category": "Categoria",
                "subcategory": "Subcategoria",
                "client": "Cliente",
                "salesman": "Vendedor",
                "sku": "SKU",
                "description": "Descripcion",
                "cluster": "Cluster",
                'ytd': "YTD", 
                'qtd': "QTD", 
                'mtd': "MTD",
                'ytg': "YTG", 
                "qtg": "QTG",
                'mtg': "MTG",
                'avg': 'AVG',
                "error": "Error",
                "abc": "ABC",
                "model": "Modelo",
            })

            new_order = [
                "Familia", "Region", "Categoria", "Subcategoria", "Cliente", "Vendedor",
                "SKU", "Descripcion", "Cluster", "YTD", "QTD", "MTD", "YTG", "QTG", 
                "MTG", "AVG", "Error", "ABC", "Modelo"
            ]

            remaining_columns = merged_df.columns.difference(new_order, sort=False)
            merged_df = merged_df[new_order + list(remaining_columns)]
            merged_df.fillna(0.0, inplace=True)

            warning = None 
            if conversion == 'price' and stock_data.exists() == False: warning ='No se encontraron datos de stock para calcular el forecast valorizado'

            if order is not None and export is None:
                merged_df = merged_df.sort_values(by=str(order), ascending=order_type)
            
            if export is not None:
                return Response(data=merged_df.to_dict(orient='records'), status=status.HTTP_200_OK)

            else:
                return Response({
                    'data': merged_df.to_dict(orient='records'),
                    'total_pages': paginator.num_pages,
                    'current_page': page.number,
                    'has_next': page.has_next(),
                    'has_previous': page.has_previous(),
                    'warning': warning
                }, status=status.HTTP_200_OK)


    class ErrorReportsView(APIView):
        permission_classes = [IsAuthenticated]
        @staticmethod
        def apply_product_filters(queryset, request):
            """
            Aplica los filtros de producto al queryset según los parámetros de la solicitud,
            asegurando que funcionen correctamente de manera anidada.
            """
            filters = Q()

            # Listar los filtros disponibles
            filter_fields = [
                ('family', 'family__in'),
                ('salesman', 'salesman__in'),
                ('region', 'region__in'),
                ('client', 'client__in'),
                ('category', 'category__in'),
                ('subcategory', 'subcategory__in'),
                ('sku', 'sku__in'),
            ]

            # Aplicar filtros de manera dinámica
            for field, query in filter_fields:
                values = request.GET.getlist(field)
                if values:
                    filters &= Q(**{query: values})

            # Filtrar el queryset con las condiciones acumuladas
            return queryset.filter(filters).distinct()
        
        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            date = request.GET.get('date')
            client = request.user.userinformation.get().client 

            try:
                scenario = Scenario.objects.get(project__client=client, name=scenario_name) if scenario_name else Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

            products = Product.objects.filter(avg__gte=0 if scenario.filter_products else -1, file__project__client=client, file__project__name=project).select_related('file')
            products = self.apply_product_filters(queryset=products, request=request)

            models = MetricsScenarios.objects.filter(scenario=scenario, best_model=True, product__in=products).values_list('model')
            model_percentages = {model: round((count / len(products)) * 100, 2) for model, count in Counter([model[0] for model in models]).items()}

            latest_dates = list(
                Sales.objects.filter(product__file__project__client=client, product__file__project__name=project)
                .values_list('date', flat=True)
                .distinct()
                .order_by('-date')[:12]
            )

            predicted = PredictedSale.objects.filter(product__in=products, best_model=True, scenario=scenario, date__in=latest_dates).values('product_id', 'date', 'sale')
            actual = Sales.objects.filter(product__in=products, date__in=latest_dates).values('product_id', 'date', 'sale')

            predicted_dict = {(pred['product_id'], pred['date']): pred['sale'] for pred in predicted}
            errors_by_date = {}

            error_function = None
            if scenario.error_type == 'MAPE':
                error_function = Error.calculate_mape
            elif scenario.error_type == 'SMAPE':
                error_function = Error.calculate_smape
            elif scenario.error_type == 'RMSE':
                error_function = Error.calculate_rmse
            elif scenario.error_type == 'MAE':
                error_function = Error.calculate_mae

            for actual_obj in actual:
                product_id, date, actual_value = actual_obj['product_id'], actual_obj['date'], actual_obj['sale']
                predicted_value = predicted_dict.get((product_id, date))
                if predicted_value is not None:
                    error = error_function(actual_value, predicted_value)
                    errors_by_date.setdefault(date, []).append(error)

            average_errors_by_date = {str(date): round(sum(errors) / len(errors), 2) for date, errors in errors_by_date.items()}
            periods_error = "Todos los periódos" if scenario.error_p == 0 else f"Últimos {scenario.error_p} periodos" 

            error_data = MetricsScenarios.objects.filter(
                scenario=scenario, 
                best_model=True, 
                product__in=products
            ).select_related('product').values(
                Familia=F('product__family'),
                Region=F('product__region'),
                Vendedor=F('product__salesman'),
                Cliente=F('product__client'),
                Categoria=F('product__category'),
                Subcategoria=F('product__subcategory'),
                SKU=F('product__sku'),
                Descripcion=F('product__description'),
                Error=F('error'),
                Ultimo_Periodo=F('last_period_error'),
            )

            error_table = list(error_data)

            return Response({
                "models_graphic": model_percentages,
                "error_graphic": average_errors_by_date,
                "error_table": error_table,
                "periods_error": periods_error
            }, status=status.HTTP_200_OK)


class ListModelsInformationAPIView(APIView):
    permission_classes = [ IsAuthenticated ]

    def get(self, request):
        scenario_name = request.GET.get('scenario')
        project = request.GET.get('project')
        product_id = request.GET.get('product')
        client = request.user.userinformation.get().client 
        
        try:
            if scenario_name:
                scenario = Scenario.objects.get(project__client=client, name=scenario_name)
            else:
                scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
        except Scenario.DoesNotExist:
            return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)
        
        predicted_sales = PredictedSale.objects.filter(product__avg__gte=0 if scenario.filter_products else -1, product__id=product_id,scenario=scenario)
        actual_sales = Sales.objects.filter(product__avg__gte=0 if scenario.filter_products else -1, product__id=product_id)
        metrics = MetricsScenarios.objects.filter(product__avg__gte=0 if scenario.filter_products else -1, product__id=product_id,scenario=scenario)

        actual_data = (
            actual_sales
            .annotate(truncated_date=TruncDate('date'))
            .values('truncated_date')
            .annotate(total_sales=Func(Sum('sale'), function='ROUND', template='%(function)s(%(expressions)s, 2)'))
            .order_by('truncated_date')
        )

        predicted_data = (
            predicted_sales
            .annotate(truncated_date=TruncDate('date'))
            .values('truncated_date', 'model')
            .annotate(total_sales=Func(Sum('sale'), function='ROUND', template='%(function)s(%(expressions)s, 2)'))
            .order_by('truncated_date')
        )

        metrics_data = (
            metrics
            .values_list('model', 'error', 'cluster', 'best_model', 'updated_at')
        ) 

        data = {
            "predicted": predicted_data,
            "actual": actual_data,
            "metrics": metrics_data
        }

        return Response(data, status=status.HTTP_200_OK)


class SetBestModelAPIView(APIView):
    def patch(self, request, *args, **kwargs):
        try:
            product_id = request.data.get("product_id")
            model_name = request.data.get("model_name")
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            client = request.user.userinformation.get().client 
        
            try:
                if scenario_name:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

            product = get_object_or_404(Product, id=product_id)
            
            new_best_model = get_object_or_404(
                MetricsScenarios,
                scenario=scenario,
                product=product,
                model=model_name
            )

            new_best_model_for_predicted_sales = PredictedSale.objects.filter(scenario=scenario, product=product, model=model_name)
        
            MetricsScenarios.objects.filter(scenario=scenario, product=product).update(best_model=False)
            PredictedSale.objects.filter(scenario=scenario, product=product).update(best_model=False)
            
            new_best_model.best_model = True
            new_best_model.updated_at = now()
            new_best_model.save()

            for predicted_sale in new_best_model_for_predicted_sales:
                predicted_sale.best_model = True
                predicted_sale.save()  # Guarda cada instancia individualmente
            
            return Response({"message": "Modelo actualizado correctamente."}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class KpisViews:

    class KpisByGroupAPIView(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            group = request.GET.get('group')
            client = request.user.userinformation.get().client

            try:
                scenario = (
                    Scenario.objects.get(project__client=client, name=scenario_name) if scenario_name
                    else Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
                )
            except Scenario.DoesNotExist:
                scenario = None

            max_historical_date = Projects.objects.get(client=client, name=project).max_historical_date

            historical_sales = Sales.objects.filter(product__file__project__client=client, product__file__project__name=project).values(f"product__{group}").annotate(
                ytg_historical=Sum(
                    'sale',
                    filter=Q(date__lte=max_historical_date) & Q(date__gte=max_historical_date - timedelta(days=365))
                ),
                qtg_historical=Sum(
                    'sale',
                    filter=Q(date__lte=max_historical_date) & Q(date__gte=max_historical_date - timedelta(days=120))
                ),
                mtg_historical=Sum(
                    'sale',
                    filter=Q(date=max_historical_date)
                ),
                ytd_historical=Sum(
                    'sale',
                    filter=Q(date__year=max_historical_date.year - 1) & Q(date__month__in=Sales.objects.filter(
                        date__year=max_historical_date.year
                    ).values_list(ExtractMonth('date'), flat=True))
                ),
                qtd_historical=Sum(
                    'sale',
                    filter=Q(date__year=max_historical_date.year - 1) & Q(date__month__in=Sales.objects.filter(
                        date__year=max_historical_date.year, 
                        date__gte=max_historical_date - timedelta(days=120),
                        date__lte=max_historical_date
                    ).values_list(ExtractMonth('date'), flat=True))
                ),
                mtd_historical=Sum(
                    'sale',
                    filter=Q(date__year=max_historical_date.year - 1) & Q(date__month=max_historical_date.month)
                ),
            )

            historical_dict = {item[f"product__{group}"]: item for item in historical_sales}

            growth_rates = {}

            if scenario:
                predicted_sales = PredictedSale.objects.filter(
                    product__file__project__client=client, 
                    product__file__project__name=project, 
                    scenario=scenario, best_model=True, 
                    date__gt=max_historical_date,
                    product__avg__gte=0 if scenario.filter_products else -1
                ).values(f"product__{group}").annotate(
                    ytg_predicted=Sum(
                        'sale',
                        filter=Q(date__gt=max_historical_date) & Q(date__lte=max_historical_date + timedelta(days=365))
                    ),
                    qtg_predicted=Sum(
                        'sale',
                        filter=Q(date__gt=max_historical_date) & Q(date__lte=max_historical_date + timedelta(days=120))
                    ),
                    mtg_predicted=Sum(
                        'sale',
                        filter=Q(date=max_historical_date + timedelta(days=31))
                    ),
                )
                
                predicted_dict = {item[f"product__{group}"]: item for item in predicted_sales}

                for category in set(historical_dict.keys()).union(predicted_dict.keys()):
                    historical = historical_dict.get(category, {})
                    predicted = predicted_dict.get(category, {})

                    growth_rates[category] = {
                        "YTG Growth": self.calculate_growth(historical.get("ytg_historical"), predicted.get("ytg_predicted")),
                        "QTG Growth": self.calculate_growth(historical.get("qtg_historical"), predicted.get("qtg_predicted")),
                        "MTG Growth": self.calculate_growth(historical.get("mtg_historical"), predicted.get("mtg_predicted")),
                        "YTD Growth": self.calculate_growth(historical.get("ytd_historical"), historical.get("ytg_historical")),
                        "QTD Growth": self.calculate_growth(historical.get("qtd_historical"), historical.get("qtg_historical")),
                        "MTD Growth": self.calculate_growth(historical.get("mtd_historical"), historical.get("mtg_historical")),
                    }
            else:
                for category in historical_dict.keys():
                    historical = historical_dict.get(category, {})

                    growth_rates[category] = {
                        "YTG Growth": "N/A",
                        "QTG Growth": "N/A",
                        "MTG Growth": "N/A",
                        "YTD Growth": self.calculate_growth(historical.get("ytd_historical"), historical.get("ytg_historical")),
                        "QTD Growth": self.calculate_growth(historical.get("qtd_historical"), historical.get("qtg_historical")),
                        "MTD Growth": self.calculate_growth(historical.get("mtd_historical"), historical.get("mtg_historical")),
                    }

            return Response(growth_rates)

        def calculate_growth(self, historical, predicted):
            if historical is None or predicted is None or historical == 0:
                return "N/A"
            return round(((predicted - historical) / historical) * 100, 2)


    class KpisByYearAPIView(APIView):
        permission_classes = [IsAuthenticated]

        def get(self, request):
            scenario_name = request.GET.get('scenario')
            project = request.GET.get('project')
            month = int(request.GET.get('month'))
            group = request.GET.get('group')
            client = request.user.userinformation.get().client 

            try:
                if scenario_name is not None:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name, project__name=project)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)
            
            if scenario is not None:
                products = Product.objects.filter(avg__gte=0 if scenario.filter_products else -1, file__project__client=client, file__project__name=project)
            
            else:
                products = Product.objects.filter(file__project__client=client, file__project__name=project)

            if group in ['family', 'region', 'salesman', 'client', 'category', 'subcategory', 'sku']:
                sales = Sales.objects.filter(
                    product__in=products,
                    date__month=month
                )

                grouped_sales = sales.values('product__' + group).annotate(
                    year=ExtractYear('date')
                ).values('product__' + group, 'year').annotate(
                    total_sales=Sum('sale')
                ).order_by('product__' + group, 'year')

                years_in_sales = grouped_sales.values_list('year', flat=True)

                predicted_sales = PredictedSale.objects.filter(
                    product__in=products,
                    date__month=month,
                    scenario=scenario, 
                    best_model=True
                ).annotate(
                    year=ExtractYear('date')
                ).exclude(year__in=years_in_sales)
                
                grouped_predicted_sales = predicted_sales.values('product__' + group).annotate(
                    year=ExtractYear('date')
                ).values('product__' + group, 'year').annotate(
                    total_sales=Sum('sale')
                ).order_by('product__' + group, 'year')

                combined_sales = list(grouped_predicted_sales) + list(grouped_sales)

                return Response(combined_sales, status=status.HTTP_200_OK)
            
            else:
                return Response({"error": "Invalid group parameter."}, status=status.HTTP_400_BAD_REQUEST)


## HELPERS ##
class GetScenarioNameAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        name = request.GET.get('name')
        client = request.user.userinformation.get().client
        project = request.GET.get('project')
        
        try:
            Scenario.objects.get(name=name, client=client, project__name=project)
            return Response(data={'exists': True}, status=status.HTTP_200_OK)

        except ObjectDoesNotExist:
            return Response(data={'exists': False}, status=status.HTTP_200_OK)
