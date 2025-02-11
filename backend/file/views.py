from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import File, Sales
from .serializer import FileSerializer, ProductSerializer, SalesSerializer
from projects.models import Projects
from .models import Product, ExogenousVariables
from forecasting.models import Scenario, MetricsScenarios, PredictedSale
from django.db.models.functions import ExtractYear, TruncDate
from django.db.models import Sum, Q, F, Avg
from django.db import transaction
from clients.models import BusinessRules
import numpy as np
import pandas as pd
from collections import defaultdict
import unicodedata
from scipy import stats
import threading
from django.core.paginator import Paginator
import traceback

## UPLOAD TEMPLATE VIEWS ##
class FileViewSet(viewsets.ModelViewSet):
    queryset = File.objects.all()
    serializer_class = FileSerializer
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        project_name = request.GET.get('project')
        project = Projects.objects.filter(name=project_name).first()
        user = request.user
        queryset = self.get_queryset().filter(uploaded_by=user, project=project.id)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        return Response(serializer.data)

    def partial_update(self, request, *args, **kwargs):
        kwargs['partial'] = True
        return self.update(request, *args, **kwargs)

    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()

        if instance.file_type == 'discontinued_data':
            products = Product.objects.filter(file__project=instance.project, discontinued=True)
            products.update(discontinued=False)

        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)


## PRODUCT VIEWS ##
class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer
    permission_classes = [IsAuthenticated]
    
    @staticmethod
    def apply_product_filters(queryset, request):
        """
        Aplica los filtros de producto al queryset según los parámetros de solicitud.
        """
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

    def list(self, request, *args, **kwargs):
        projectname = request.GET.get('project')
        page_number = request.GET.get('page', '1')
        order = request.GET.get('order_by')  # Campo para ordenar
        order_type = False if request.GET.get('order_type') == 'desc' else True  # Tipo de orden (ascendente o descendente)
        client = request.user.userinformation.get().client

        # Obtener los productos del proyecto
        query = self.get_queryset().filter(file__project__name=projectname, file__project__client=client)

        # Aplicar filtros de productos
        query = self.apply_product_filters(query, request)

        # Ordenar productos
        if order:
            order_query = f"-{order}" if not order_type else order
            query = query.order_by(order_query)

        if page_number == 'all':  # Si el parámetro `page` es 'all', devolver todos los resultados
            serializer = self.get_serializer(query, many=True)
            response_data = {
                "total_items": query.count(),
                "products": serializer.data,
            }

        else:
            # Paginación
            paginator = Paginator(query, 10)  # 10 productos por página
            page = paginator.get_page(int(page_number))

            # Serializar los productos de la página solicitada
            serializer = self.get_serializer(page.object_list, many=True)

            # Preparar datos de respuesta con información de paginación
            response_data = {
                "total_items": paginator.count,
                "total_pages": paginator.num_pages,
                "current_page": page_number,
                "products": serializer.data,
            }

        return Response(response_data, status=status.HTTP_200_OK)

    def retrieve(self, request, *args, **kwargs):
        product = self.get_object()  
        scenario_name = request.GET.get('scenario')
        project = request.GET.get('project')
        type_of_retrieve = request.GET.get('type')  
        client = request.user.userinformation.get().client
        avg = product.avg 
        
        response_data = {}
        sales_data = Sales.objects.filter(product=product).first()

        if sales_data is None:
            return Response({'message': 'No se encontraron historial de ventas para este producto'}, status=status.HTTP_400_BAD_REQUEST)

        if type_of_retrieve == 'both':

            try:
                if scenario_name:
                    scenario = Scenario.objects.get(project__client=client, name=scenario_name)
                else:
                    scenario = Scenario.objects.filter(project__client=client, project__name=project).order_by('-created_at').first()
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)
                
            if (scenario.filter_products is True and avg > 0) or scenario.filter_products is False:
                metrics = MetricsScenarios.objects.get(
                    scenario=scenario,
                    product=product,
                    best_model=True
                ) 

                if metrics:
                    response_data['metrics'] = {
                        'error': metrics.error,
                        'last_period_error': metrics.last_period_error,
                        'ytg': metrics.ytg,
                        'qtg': metrics.qtg,
                        'mtg': metrics.mtg
                    }

                predicted_sales = PredictedSale.objects.filter(
                    scenario=scenario,
                    product=product,
                    model=metrics.model
                ).values('date', 'sale').order_by('date')
                
                if predicted_sales:
                    predicted_sales_data = [
                        {'date': sale['date'], 'sale': sale['sale']}
                        for sale in predicted_sales
                    ]
                    response_data['predicted_sales'] = predicted_sales_data
            
            else:
                response_data['predicted_sales'] = []
                response_data['metrics'] = {}
                response_data['warning'] = f'Este producto no se corrio en el escenario seleccionado, solo venta historica'

        sales_data = Sales.objects.filter(product=product).values('date', 'sale').order_by('date')

        if sales_data:
            sales_data_response = [
                {'date': sale['date'], 'sale': sale['sale']}
                for sale in sales_data
            ]
            response_data['historical_sales'] = sales_data_response
            response_data['kpis'] = {
                'ytd': product.ytd,
                'qtd': product.qtd,
                'mtd': product.mtd
            }

        product_serializer = self.get_serializer(product)
        response_data['product'] = product_serializer.data

        return Response(response_data, status=status.HTTP_200_OK)

    def partial_update(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer, status=status.HTTP_200_OK)

        return Response(serializer.data)

    def perform_update(self, serializer):
        serializer.save()


## EXPLORATION GRAPHICS AND TABLE VIEW #
class HistoricalSalesAPIView(APIView):
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

    def aggregate_sales(self, sales_data, period):
        if period == 'linegraph':
            return (
                sales_data
                .annotate(truncated_date=TruncDate('date'))
                .values('truncated_date')
                .annotate(total_sales=Sum('sale'))
                .order_by('truncated_date')
            )
            
        elif period == 'bargraph':
            return (
                sales_data
                .annotate(year=ExtractYear('date'))
                .values('year')
                .annotate(total_sales=Sum('sale'))
                .order_by('year')
            )

    def generate_table_data(self, sales_data):
        sales = sales_data.values(
            'product_id', 'product__family', 'product__region',
            'product__client', 'product__salesman',
            'product__category', 'product__subcategory',
            'product__sku', 'date'
        ).annotate(sale=Sum('sale'))

        return sales

    def get(self, request):
        project = request.GET.get('project')
        type_of = request.GET.get('typeofget')
        client = request.user.userinformation.get().client 

        # Aplicar filtros a productos
        products = Product.objects.filter(file__project__client=client, file__project__name=project)
        products = self.apply_product_filters(queryset=products, request=request)

        sales_data = Sales.objects.filter(product__in=products).select_related('product')

        if type_of in ['linegraph', 'bargraph']:
            response_data = self.aggregate_sales(sales_data, type_of)
        
        else:
            response_data = self.generate_table_data(sales_data=sales_data)

        return Response(response_data, status=status.HTTP_200_OK)


class GraphicOutliersAPIView(APIView):
    permission_classes = [IsAuthenticated]

    @staticmethod
    def apply_product_filters(queryset, request):
        """
        Aplica los filtros de producto al queryset según los parámetros de la solicitud,
        asegurando que funcionen correctamente de manera anidada.
        """
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
        project = request.GET.get("project")  # Proyecto al cual pertenecen los productos
        client = request.user.userinformation.get().client

        # Obtener el queryset de productos y aplicar los filtros
        product_queryset = Product.objects.filter(file__project__client=client, file__project__name=project)
        filtered_products = self.apply_product_filters(product_queryset, request)

        # Obtener las ventas históricas para los productos filtrados
        sales = Sales.objects.filter(
            product__in=filtered_products
        ).values('date', 'sale').order_by('date')

        # Convertir los datos en un DataFrame de pandas
        sales_data = list(sales)
        df_sales = pd.DataFrame(sales_data)

        if df_sales.empty:
            return Response(
                data={"error": "No sales data found for the selected project and filters."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Agrupar las ventas por fecha y calcular la suma
        df_sales_grouped = df_sales.groupby('date')['sale'].sum().reset_index()

        # Calcular Z-scores para detectar outliers
        z_scores = stats.zscore(df_sales_grouped['sale'])
        abs_z_scores = np.abs(z_scores)

        # Filtrar los datos donde el Z-score absoluto es mayor que 1.5 (outliers)
        filtered_entries = abs_z_scores > 1.5
        df_filtered = df_sales_grouped[filtered_entries]

        # Preparar la respuesta
        return Response({
            "dates": df_sales_grouped['date'].tolist(),
            "sales": df_sales_grouped['sale'].tolist(),
            "outliers": df_filtered['date'].tolist()
        }, status=status.HTTP_200_OK)


## SALES VIEW ##
class SalesViewSet(viewsets.ModelViewSet):
    queryset = Sales.objects.all()
    serializer_class = SalesSerializer
    permission_classes = [IsAuthenticated]
    
    def list(self, request, *args, **kwargs):
        projectname = request.GET.get('project')
        client = request.user.userinformation.get().client 
        query = self.get_queryset().filter(product__file__project__client=client, product__file__project__name=projectname)
        serializer = self.get_serializer(query, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


## PRODUCT FILTERS ##
class ProductFiltersAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        project_name = request.GET.get('project')
        group = request.GET.get('group')
        client = request.user.userinformation.get().client 

        # Obtener los productos base para el proyecto
        products = Product.objects.filter(file__project__client=client, file__project__name=project_name)

        # Aplicar filtros actuales
        filters = Q()

        family_filters = request.GET.getlist('family')
        if family_filters:
            filters &= Q(family__in=family_filters)

        client_filters = request.GET.getlist('client')
        if client_filters:
            filters &= Q(client__in=client_filters)

        region_filters = request.GET.getlist('region')
        if region_filters:
            filters &= Q(region__in=region_filters)

        category_filters = request.GET.getlist('category')
        if category_filters:
            filters &= Q(category__in=category_filters)

        subcategory_filters = request.GET.getlist('subcategory')
        if subcategory_filters:
            filters &= Q(subcategory__in=subcategory_filters)

        salesman_filters = request.GET.getlist('salesman')
        if salesman_filters:
            filters &= Q(salesman__in=salesman_filters)

        sku_filters = request.GET.getlist('sku')
        if sku_filters:
            filters &= Q(sku__in=sku_filters)

        # Aplicar los filtros acumulados al queryset
        filtered_products = products.filter(filters)

        if group:
            valid_groups = [
                "family", "client", "region", "category",
                "subcategory", "salesman", "sku", "description"
            ]
            if group in valid_groups:
                group_values = filtered_products.values_list(group, flat=True).distinct()
                return Response(list(group_values), status=status.HTTP_200_OK)
            else:
                return Response(
                    {"error": "Invalid group specified."},
                    status=status.HTTP_400_BAD_REQUEST
                )

        # Calcular los valores únicos basados en el queryset filtrado
        families = filtered_products.values_list('family', flat=True).distinct().order_by('-family')
        clients = filtered_products.values_list('client', flat=True).distinct().order_by('-client')
        regions = filtered_products.values_list('region', flat=True).distinct().order_by('-region')
        categories = filtered_products.values_list('category', flat=True).distinct().order_by('-category')
        subcategories = filtered_products.values_list('subcategory', flat=True).distinct().order_by('-subcategory')
        salesmen = filtered_products.values_list('salesman', flat=True).distinct().order_by('-salesman')
        skus = filtered_products.values_list('sku', flat=True).distinct().order_by('-sku')

        # Construir el resultado
        result = {
            "Familia": list(families),
            "Cliente": list(clients),
            "Region": list(regions),
            "Categoria": list(categories),
            "Subcategoria": list(subcategories),
            "Vendedor": list(salesmen),
            "SKU": list(skus)
        }

        result = {key: value for key, value in result.items() if not (len(value) == 1 and value[0] == "")}

        return Response(result, status=status.HTTP_200_OK)
    

## UPDATE SALES CSV ##
class UploadSalesCSV(APIView):
    permission_classes = [IsAuthenticated]

    @staticmethod
    def format_dates(df: pd.DataFrame):
        date_cols = df.columns[8:] 
        date_cols_format = pd.to_datetime(date_cols, format='%d/%m/%y', errors='coerce')
        df.rename(columns=dict(zip(date_cols, date_cols_format.strftime('%Y-%m-%d'))), inplace=True)

        return df

    @staticmethod
    def update_product_avg_after_bulk(to_create, to_update):
        product_ids_to_update = set(
            Sales.objects.filter(id__in=[sale.id for sale in to_create + to_update])
            .values_list('product', flat=True)
        )

        products_to_update = Product.objects.filter(id__in=product_ids_to_update)

        for product in products_to_update:
            avg_sale = Sales.objects.filter(product=product) \
                .order_by('-date')[:12] \
                .aggregate(average=Avg('sale'))['average'] or 0.0

            product.avg = avg_sale
        
        Product.objects.bulk_update(products_to_update, ['avg'], batch_size=10000)

    def post(self, request, *args, **kwargs):
        try:
            sales_data = request.data.get('data')
            project = request.data.get('project')
            client = request.user.userinformation.get().client 

            if not sales_data:
                return Response({"error": "no_data_provided"}, status=status.HTTP_400_BAD_REQUEST)
            
            df_sales = pd.DataFrame(sales_data)
            to_create = []
            to_update = []
            
            df_sales = df_sales.loc[:, ~df_sales.columns.str.contains('__EMPTY')]
            project = Projects.objects.get(client=client, name=project)
            actual_max_hsd = project.max_historical_date

            df_sales = self.format_dates(df=df_sales)
            last_columnn_date = pd.to_datetime(df_sales.columns[-1], format='%Y-%m-%d').date()

            if last_columnn_date > actual_max_hsd:
                project.max_historical_date = last_columnn_date
                project.save()
            
            for column in ["Family", "Region", "SKU", "Category", "Subcategory", "Client", "Salesman", "Description"]:
                df_sales[column] = df_sales[column].apply(
                    lambda x: ''.join(
                        c for c in unicodedata.normalize('NFD', x.strip()) if not unicodedata.combining(c)
                    ) if isinstance(x, str) else x
                )
     
            if 'Product ID' in df_sales.columns and df_sales['Product ID'].notnull().any():
                df_sales.rename(columns={'Product ID': 'product_id'}, inplace=True)
                df_sales.drop(columns=["Family", "Region", "SKU", "Category", "Subcategory", "Client", "Salesman", "Description"], inplace=True)

            else:
                df_sales.rename(columns={
                    'SKU':'sku',
                    "Family": 'family', 
                    "Region": 'region',  
                    "Category": 'category', 
                    "Subcategory": 'subcategory', 
                    "Client": 'client', 
                    "Salesman": 'salesman', 
                    "Description": 'description'
                }, inplace=True) 

                fields = BusinessRules.objects.filter(client=client).first().fields

                df_sales['hash'] = df_sales.apply(lambda row: '-'.join(str(row[field]) for field in fields), axis=1)
                
                hash_to_product_id = {
                    '-'.join(str(getattr(product, field)) for field in fields): product.id
                    for product in Product.objects.filter(file__project=project, file__project__client=client)
                }

                df_sales['product_id'] = df_sales['hash'].map(hash_to_product_id)
                df_sales.dropna(subset=['product_id'], inplace=True)
                columns = ['product_id'] + [col for col in df_sales.columns if col != 'product_id']
                df_sales = df_sales[columns]
                df_sales.drop(columns=['hash'], inplace=True)

                df_sales.drop(columns=["family", "region", "sku", "category", "subcategory", "client", "salesman", "description"], inplace=True)
            
            print(df_sales)

            df_sales = df_sales.melt(
                id_vars=df_sales.columns[:1],
                value_vars=df_sales.columns[1:], 
                var_name='date', 
                value_name='sale'
            )
            
            print(df_sales)

            records = df_sales.to_dict(orient="records")

            existing_records = {
                (sale.product_id, sale.date): sale
                for sale in Sales.objects.filter(
                    product_id__in=df_sales["product_id"].unique(),
                    date__in=df_sales["date"].unique()
                )
            }

            for record in records:
                key = (int(record["product_id"]), record["date"])
                
                if key in existing_records:
                    existing_record = existing_records[key]
                    existing_record.sale = record["sale"]
                    to_update.append(existing_record)
                
                else:
                    to_create.append(Sales(product_id=record["product_id"], date=record["date"], sale=record["sale"]))

            with transaction.atomic():
                if to_create:
                    Sales.objects.bulk_create(to_create, batch_size=10000)
                if to_update:
                    Sales.objects.bulk_update(to_update, ["sale"], batch_size=10000)
            
            thread = threading.Thread(target=self.update_product_avg_after_bulk, args=(to_create, to_update))
            thread.start()

            return Response({"message": "sale_data_updated"}, status=status.HTTP_200_OK)

        except Exception as e:
            traceback.print_exc()
            print(e)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ExogenousVariablesViews:

    class ExogenousVariablesGraphicAPIView(APIView):
        permission_classes = [IsAuthenticated]

        @staticmethod
        def apply_product_filters(queryset, request):
            """
            Aplica los filtros de producto al queryset según los parámetros de la solicitud,
            asegurando que funcionen correctamente de manera anidada.
            """
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
            project = request.GET.get('project')
            client = request.user.userinformation.get().client 

            product_queryset = Product.objects.filter(file__project__client=client, file__project__name=project)
            filtered_products = self.apply_product_filters(product_queryset, request)

            # Obtener fechas únicas de ventas basadas en los productos filtrados
            dates = Sales.objects.filter(product__in=filtered_products).values_list('date', flat=True).distinct()
            sorted_dates = sorted([date.strftime('%Y-%m-%d') for date in dates])

            exog = (
                ExogenousVariables.objects
                .filter(file__project__client=client, file__project__name=project)
                .select_related('sale')
                .values('variable', 'sale__date', 'exog')
                .distinct()
                .order_by('variable', 'sale__date')
            )
            df = pd.DataFrame(list(exog))

            if df.empty:
                return Response(
                    data={"error": "No exogenous variables or sales data found for this project."},
                    status=status.HTTP_404_NOT_FOUND
                )

            distinct_variables = df['variable'].unique().tolist()

            sales_dict = defaultdict(list)

            for variable in distinct_variables:
                variable_data = df[df['variable'] == variable]

                complete_data = {date: 0 for date in sorted_dates}

                for _, row in variable_data.iterrows():
                    date_str = row['sale__date'].strftime('%Y-%m-%d')
                    complete_data[date_str] = row['exog']

                sales_dict[variable] = [complete_data[date] for date in sorted_dates]

            result = {
                "variables": distinct_variables,
                "dates": sorted_dates,
                "sales": sales_dict
            }

            return Response(data=result, status=status.HTTP_200_OK)

    class AllocationMatrixAPIView(APIView):
        permission_classes = [IsAuthenticated]
        
        @staticmethod
        def calculate_allocation_matrix(variable, historical, exogenous):
            npcorr = np.corrcoef(historical, exogenous)
            correlation = npcorr[0, 1]
            data = { variable: round(correlation, 3) if pd.notna(correlation) else 0.0 }
            return data
        
        def get(self, request):
            project = request.GET.get('project')
            client = request.user.userinformation.get().client 

            exog_data = ExogenousVariables.objects.filter(
                sale__product__file__project__client=client,
                sale__product__file__project__name=project,  # Relación desde sale a product y luego a project
                sale__product__discontinued=False           # Filtrar productos activos
            ).annotate(
                product_id=F('sale__product_id'),           # Anotar el product_id desde sale
            ).values(
                'product_id', 'exog', 'variable'           # Obtener los campos requeridos
            ).order_by('product_id')
            
            df = pd.DataFrame(list(exog_data))

            productids = df['product_id'].drop_duplicates().tolist()
            
            product_info = Product.objects.filter(id__in=productids).values('id', 'family', 'region', 'category', 'subcategory', 'client', 'salesman', 'sku')

            product_info_dict = {product['id']: product for product in product_info}  

            allocation_results = {}

            for (product_id, variable), group in df.groupby(['product_id', 'variable']):
                historical = list(Sales.objects.filter(product__id=product_id).values_list('sale', flat=True))
                
                # Obtener la lista de valores exógenos
                exogenous = group['exog'].tolist()
                
                # Asegurar que exogenous tenga la misma longitud que historical
                if len(exogenous) < len(historical):
                    exogenous.extend([0.0] * (len(historical) - len(exogenous)))
                
                result = self.calculate_allocation_matrix(variable=variable, historical=historical, exogenous=exogenous)    

                product_data = product_info_dict.get(product_id, {})
                product_details = " || ".join([
                    product_data.get('family', ''),
                    product_data.get('region', ''),
                    product_data.get('category', ''),
                    product_data.get('subcategory', ''),
                    product_data.get('client', ''),
                    product_data.get('salesman', ''),
                    product_data.get('sku', '')
                ])
                
                if product_id not in allocation_results:
                    allocation_results[product_id] = {
                        "ID": int(product_id),
                        "Producto": product_details
                    }
                
                # Actualiza el diccionario de allocation_results con los resultados
                allocation_results[product_id].update(result)  

            # Convertimos el diccionario de resultados a una lista para la respuesta
            return Response(list(allocation_results.values()), status=status.HTTP_200_OK)
