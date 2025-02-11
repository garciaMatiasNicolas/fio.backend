from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .models import Stock
from file.models import Product, Sales, File
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Avg, StdDev, F, Sum
from rest_framework.exceptions import ValidationError
import traceback
from datetime import datetime, timedelta
from scipy.special import erfinv
from rest_framework.decorators import action
from forecasting.models import Scenario, PredictedSale
from clients.models import BusinessRules
import math
from collections import defaultdict
from django.db import transaction
from .serializer import StockSerializer
from rest_framework import viewsets
import unicodedata
import csv
import pandas as pd
from django.core.paginator import Paginator


class StockCSVUpdateView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        try:
            csv_file = request.FILES.get('file')

            if not csv_file.name.endswith('.csv'):
                return Response({"error": "El archivo debe ser un CSV."}, status=status.HTTP_400_BAD_REQUEST)

            data = csv.DictReader(csv_file.read().decode('utf-8').splitlines())
            updates = []

            with transaction.atomic():  # Manejo transaccional para garantizar integridad
                for row in data:
                    product_id = row.get('product_id')
                    if not product_id:
                        continue  # Ignorar filas sin product_id

                    try:
                        stock = Stock.objects.get(product__id=product_id)
                        for field, value in row.items():
                            if field != 'product_id' and hasattr(stock, field):
                                setattr(stock, field, value if value != '' else None)

                        stock.save()
                        updates.append(product_id)
                    
                    except Stock.DoesNotExist:
                        continue  

            return Response({"message": f"Stock actualizado para {len(updates)} productos.", "updated_products": updates})

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class ReaproAPIView(APIView):
    permission_classes = [IsAuthenticated]

    @staticmethod
    def calculate_optimal_batch(c, d, k):
        c = c if c >= 0 else 0 * 360
        d = int(d)/100
        k = float(f'0.{k}')
        EOQ = math.sqrt((2 * c * d) / k)
        return EOQ
    
    @staticmethod
    def calculate_traffic_light(products: list):
        try:
            count_articles = defaultdict(int)
            sum_sales = defaultdict(float)
            sum_stock = defaultdict(float)
            sum_valued_sales = defaultdict(int)
            sum_valued_stock = defaultdict(int)
            sum_overflow = defaultdict(int)

            for product in products:
                avg_sales = float(product["Venta diaria histórico"])
                caracterizacion = product["Caracterizacion"]
                count_articles[caracterizacion] += 1
                sum_sales[caracterizacion] += avg_sales
                sum_stock[caracterizacion] += float(product["Stock"])
                sum_valued_sales[caracterizacion] += int(product["Venta valorizada"])
                sum_valued_stock[caracterizacion] += int(product["Valorizado"])
                sum_overflow[caracterizacion] += int(product["Sobrante valorizado"])

            result = [
                {
                    "Caracterizacion": key,
                    "Cantidad de productos": count_articles[key], 
                    "Suma venta diaria": sum_sales[key],
                    "Suma de stock": sum_stock[key],
                    "Venta valorizada": sum_valued_sales[key],
                    "Stock valorizado": sum_valued_stock[key],
                    "Sobrante valorizado": sum_overflow[key],
                }
                for key in count_articles
            ]

            total_count_articles = sum(count_articles.values())
            total_sum_sales = sum(sum_sales.values())
            total_sum_stock = sum(sum_stock.values())
            total_valued_sales = sum(sum_valued_sales.values())
            total_valued_stock = sum(sum_valued_stock.values())
            total_overflow = sum(sum_overflow.values())

            result.append({
                "Caracterizacion": "Suma total",
                "Cantidad de productos": total_count_articles,
                "Suma venta diaria": total_sum_sales,
                "Suma de stock": total_sum_stock,
                "Venta valorizada": total_valued_sales,
                "Stock valorizado": total_valued_stock,
                "Sobrante valorizado":total_overflow,
            })

            sorted_results = sorted(result, key=lambda item: item["Caracterizacion"])

            return sorted_results

        except Exception as err:
            print("ERROR EN SEMÁFORO:", err)

    @staticmethod
    def calculate_safety_stock(data: list):
        try:
            final_data = []
            for product in data:
                avg_sales_per_day = product['avg_sales_per_day_historical']
                desv_per_day = product['desv_per_day_historical']
                lead_time = product['lead_time']
                service_level = product['service_level'] / 100
                desv_est_lt_days = product['desv_std']

                service_level_factor = round(erfinv(2 * service_level - 1) * 2**0.5, 2)

                try:
                    desv_comb = round(
                        ((lead_time * desv_per_day * desv_per_day) + 
                        (avg_sales_per_day * avg_sales_per_day * desv_est_lt_days * desv_est_lt_days)) ** 0.5,
                        2
                    )
                except OverflowError:
                    desv_comb = 0 

                safety_stock_units = round(service_level_factor * desv_comb, 2)
                if not (safety_stock_units > 0):  
                    safety_stock_units = 0

                reorder_point = round(lead_time * avg_sales_per_day + safety_stock_units, 2)
                if not (reorder_point > 0):  
                    reorder_point = 0.0

                safety_stock_days = round(safety_stock_units / avg_sales_per_day, 2) if avg_sales_per_day > 0 else 0

                safety_stock = {
                    'Product ID': product['product_id'],
                    'Familia': product['family'],
                    'Categoria': product['category'],
                    'Vendedor': product['salesman'],
                    'Subcategoria': product['subcategory'],
                    'Cliente': product['client'],
                    'Region': product['region'],
                    'SKU': product['sku'],
                    'Descripcion': product['description'],
                    'Promedio': str(avg_sales_per_day),
                    'Desviacion': str(desv_per_day),
                    'Coeficiente desviacion': str(round(float(avg_sales_per_day) / float(desv_per_day), 2)) if float(desv_per_day) != 0 else "0",
                    'Tiempo demora': str(lead_time),
                    'Variabilidad demora': str(desv_est_lt_days),
                    'Nivel servicio': str(service_level),
                    'Factor Nivel Servicio': str(service_level_factor),
                    'Desviacion combinada': str(desv_comb),
                    'Punto reorden': str(reorder_point),
                    'Stock Seguridad (días)': str(int(round(safety_stock_days))),
                    'Stock Seguridad (unidad)': str(int(round(safety_stock_units)))
                }

                final_data.append(safety_stock)

            return final_data
        except Exception as err:
            print("ERROR EN CALCULO STOCK SEGURIDAD", err)

    @staticmethod
    def calculate_drp(products: list, is_forecast: bool):
        try:
            def round_up(n, dec):
                try:
                    factor = n / dec
                except ZeroDivisionError:
                    factor = 0

                factor = round(factor)
                return factor * dec

            available_stock = 0
            avg_sales_per_day = 0

            coverage_by_sku_region = {}

            for product in products:
                sku = product['SKU']
                available_stock += float(product["Stock disponible"])
                avg_sales_per_day += float(product[f"Venta diaria {'predecido' if is_forecast else 'histórico'}"])
                region = product['Region']
                coverage = int(product['Cobertura (días)'])


                if sku not in coverage_by_sku_region:
                    coverage_by_sku_region[sku] = {}

                coverage_by_sku_region[sku][region] = {
                    "cobertura": coverage,
                    "stock_disponible": float(product["Stock disponible"]),
                    "stock_fisico": float(product["Stock"])
                }

            transformed_products = []
            for product in products:
                new_product = product.copy()
                transformed_products.append(new_product)

            for product in transformed_products:
                lead_time_drp = int(round(float(product["Demora en dias (DRP)"])))
                avg_sales = float(product[f"Venta diaria {'predecido' if is_forecast else 'histórico'}"])
                coverage_in_days = int(product["Cobertura (días)"])
                reorder_point_drp = int(round(float(product["Stock de seguridad (DRP)"]))) + lead_time_drp
                replenish = "Si" if coverage_in_days < reorder_point_drp else "No"
                next_coverage_in_days = int(product['Cobertura prox. compra (días)'])
                lead_time = float(product['Demora en dias'])
                safety_stock_drp = float(product['Stock de seguridad (DRP)'])
                calc = (lead_time_drp + next_coverage_in_days + lead_time + safety_stock_drp - coverage_in_days) * avg_sales
                how_much_drp = 0 if replenish == "No" else float(math.ceil(calc))

                try:
                    product["Cobertura Total"] = int(round(available_stock / avg_sales_per_day))
                except ZeroDivisionError:
                    product["Cobertura Total"] = 0

                product["Punto de reorden (DRP)"] = reorder_point_drp
                product["¿Repongo?"] = replenish
                product["¿Cuanto repongo?"] = how_much_drp
                product["Activar"] = "Activo" if int(product["Cobertura (Stock Físco)"]) < reorder_point_drp and coverage_in_days > int(product["Cobertura (Stock Físco)"]) else "" 

                if replenish == "Si":
                    regions_with_sufficient_coverage = {
                        reg: coverage_by_sku_region[product['SKU']][reg]['cobertura']
                        for reg in coverage_by_sku_region[product['SKU']]
                        if coverage_by_sku_region[product['SKU']][reg]['cobertura'] >= reorder_point_drp and
                        coverage_by_sku_region[product['SKU']][reg]['stock_fisico'] > how_much_drp
                    }

                    if len(regions_with_sufficient_coverage) == 0:
                        product["Distribuir desde"] = "Comprar"
                    elif len(regions_with_sufficient_coverage) == 1:
                        product["Distribuir desde"] = next(iter(regions_with_sufficient_coverage))
                    else:
                        product["Distribuir desde"] = max(regions_with_sufficient_coverage, key=regions_with_sufficient_coverage.get)

                else:
                    product["Distribuir desde"] = "No repone"

            return transformed_products

        except Exception as e:
            print(f"Error en el cálculo del DRP: {e}")
            return []

    def calculate_reapro(self, data: list, next_buy_days: int, is_forecast: bool, d: int, k: int):
        try:
            def round_up(n, dec):
                factor = n / dec
                factor = round(factor)
                return factor * dec

            results = []
            
            for item in data:
                avg_sales_historical = round(item["avg_sales_per_day_historical"], 2)
                cost_price = item["cost_price"]
                price = item['price']
                avg_sales_forecast = round(item["avg_sales_per_day_forecast"], 2) if is_forecast else 0.0
                purchase_order = item['purchase_order']
                avg_sales = item[f'avg_sales_per_day_{"forecast" if is_forecast else "historical"}']
                stock = item["stock"]
                available_stock = item['stock'] - item['sales_order_pending_delivery'] + purchase_order
                lead_time = item['lead_time']
                safety_stock = item['safety_stock']
                reorder_point = next_buy_days + lead_time + safety_stock
                days_of_coverage = round(available_stock / avg_sales) if avg_sales != 0 else 9999
                buy = 'Si' if (days_of_coverage - reorder_point) < 1 else 'No'
                optimal_batch = item["eoq"]
                overflow_units = stock if avg_sales == 0 else (0 if (stock / avg_sales) - reorder_point < 0 else round((stock / avg_sales - reorder_point)*avg_sales)) 
                overflow_price = round(overflow_units*cost_price)
                lot_sizing = item['lot_sizing']
                sales_order = item['sales_order_pending_delivery']
                is_obs = item['slow_moving']
                purchase_unit = item['purchase_unit']
                make_to_order = item['make_to_order']
                coverage_stock = round(int((round(stock))) / avg_sales, 2) if avg_sales != 0 else 0

                try:
                    next_buy = datetime.now() + timedelta(days=days_of_coverage - lead_time) if days_of_coverage != 0 \
                        else datetime.now()

                except OverflowError:
                    next_buy = ""

                if days_of_coverage == 9999:

                    stock_status = "Obsoleto"
                    if available_stock != 0:
                        characterization = "0-Con stock sin ventas"
                    else:
                        characterization = "Sin stock"

                elif days_of_coverage > 360:
                    stock_status = 'Alto sobrestock'
                    characterization = "1-Más de 360 días"

                elif days_of_coverage > 180:
                    stock_status = 'Sobrestock'
                    characterization = "2-Entre 180 y 360"
                    
                elif days_of_coverage > 30:

                    stock_status = 'Normal'
                    if days_of_coverage > 90:
                        characterization = "3-Entre 90 y 180"
                    else:
                        characterization = "4-Entre 30 y 90"

                elif days_of_coverage > 15:
                    stock_status = 'Riesgo quiebre'
                    characterization = "5-Entre 15 y 30"

                elif days_of_coverage >= 0:
                    stock_status = "Quiebre"
                    characterization = "6-Menos de 15"
                
                else:
                    stock_status = 'Stock negativo'
                    
                    if available_stock != 0:
                        characterization = "0-Con stock sin ventas"
                    else:
                        characterization = "Sin stock"
    
                next_buy = next_buy.strftime('%Y-%m-%d') if isinstance(next_buy, datetime) else next_buy
                final_buy = ('Si' if available_stock - sales_order + purchase_order < 0 else 'No') if make_to_order == 'MTO' else buy
                
                if final_buy == "Si":
                    
                    if is_obs == 'OB':
                        how_much = 0
                        how_much_vs_lot_sizing = 0
                        how_much_purchase_unit = 0
                    
                    else:
                        if make_to_order == "MTO" or (available_stock > 0 >= avg_sales):
                            how_much = abs(available_stock) if available_stock < 0 else 0
                        
                        else: 
                            how_much = max(optimal_batch, (next_buy_days + lead_time + safety_stock - days_of_coverage) * avg_sales )

                        how_much_vs_lot_sizing = round_up(how_much, int(lot_sizing)) if int(lot_sizing) != 0.0 else how_much
                        how_much_vs_lot_sizing = max(how_much_vs_lot_sizing, optimal_batch)

                        if make_to_order == "MTO":
                            how_much_vs_lot_sizing = abs(available_stock) if available_stock < 0 else 0
                        
                        else: 
                            how_much_vs_lot_sizing = round(how_much_vs_lot_sizing)
                        
                        how_much_purchase_unit = round(how_much_vs_lot_sizing * purchase_unit)
                
                else:
                    how_much = 0
                    how_much_vs_lot_sizing = 0
                    how_much_purchase_unit = 0
                
                valued_cost = round(cost_price*how_much_vs_lot_sizing,2)

                optimal_batch_calc = self.calculate_optimal_batch(c=avg_sales, d=d, k=k)
                
                try:
                    thirty_days = days_of_coverage - 30 + round(how_much) / avg_sales
                    
                    if thirty_days < reorder_point:
                        thirty_days = avg_sales * 30
                    
                    else:
                        thirty_days = 0

                except:
                    thirty_days = 0
                
                try:
                    sixty_days = days_of_coverage - 60 + round(how_much) / avg_sales + thirty_days / avg_sales 
                    
                    if sixty_days < reorder_point:
                       sixty_days = avg_sales * 30
                    
                    else:
                        sixty_days = 0
                
                except:
                    sixty_days = 0

                try:
                    ninety_days = days_of_coverage - 90 + round(how_much) / avg_sales + thirty_days / avg_sales + sixty_days / avg_sales

                    if ninety_days < reorder_point:
                       ninety_days = avg_sales * 30
                    
                    else:
                        ninety_days = 0
                
                except:
                    ninety_days = 0

                try:
                    drp_lead_time = item["drp_lead_time"]
                    drp_safety_stock = item["drp_safety_stock"]
                    drp_lot_sizing = item["drp_lot_sizing"]
                
                except KeyError:
                    drp_lead_time = "Falta de subir en Stock data"
                    drp_safety_stock = "Falta de subir en Stock data"
                    drp_lot_sizing = "Falta de subir en Stock data"

                try:
                    client_sku = str(item["Supplier SKU code"])
                except KeyError:
                    client_sku = ""

                if item['sales_info'] is False:
                    data = 'Producto sin información de Venta'
                
                elif item['stock_info'] is False:
                    data = 'Producto sin información de Stock'
                
                elif item['forecast_info'] is False:
                    data = 'Producto sin información de Forecast'
                
                else:
                    data = 'Producto con stock y ventas'
                    

                stock = {
                    'Data': data,
                    'ID': item['product_id'],
                    'Familia': item['family'],
                    'Categoria': item['category'],
                    'Vendedor': item['salesman'],
                    'Subcategoria': item['subcategory'],
                    'Cliente': item['client'],
                    'Region': item['region'],
                    'SKU': item['sku'],
                    'Descripcion': item['description'],
                    'Stock': int(round(stock)),
                    'Stock disponible': int(round(available_stock)),
                    'Lote mínimo de compra': optimal_batch,
                    'Ordenes de venta pendientes': sales_order,
                    'Ordenes de compra': purchase_order,
                    'Venta diaria histórico': avg_sales_historical,
                    'Venta diaria predecido': avg_sales_forecast,
                    'Cobertura (Stock Físco)': coverage_stock,
                    'Cobertura (días)': str(days_of_coverage),
                    'Punto de reorden': str(reorder_point),
                    '¿Compro?': str(final_buy) if is_obs != 'OB' else 'No',
                    '¿Cuanto?': round(how_much),
                    '¿Cuanto? (Lot Sizing)': round(how_much_vs_lot_sizing),
                    '¿Cuanto? (Purchase Unit)': how_much_purchase_unit,
                    'Compra 30 días':  0 if make_to_order == "MTO" or is_obs == "OB" else thirty_days,
                    'Compra 60 días' : 0 if make_to_order == "MTO" or is_obs == "OB" else sixty_days,
                    'Compra 90 días': 0 if make_to_order == "MTO" or is_obs == "OB" else ninety_days,
                    'Estado': str(stock_status),
                    'Cobertura prox. compra (días)': str(days_of_coverage - next_buy_days),
                    'Stock seguridad en dias': str(safety_stock),
                    'Unidad de compra': purchase_unit,
                    'Lote de compra': lot_sizing,
                    'EOQ (Calculado)': optimal_batch_calc,
                    'Precio unitario': round(price),
                    "Costo del producto": round(cost_price),
                    'MTO': make_to_order if make_to_order == 'MTO' else '',
                    'OB': is_obs if is_obs == 'OB' else '',
                    'XYZ': item['xyz'],
                    'ABC Cliente': item['abc'],
                    'Sobrante valorizado': int(round(overflow_price)),
                    'Demora en dias (DRP)': drp_lead_time,
                    'Stock de seguridad (DRP)': drp_safety_stock,
                    'Lote de compra (DRP)': drp_lot_sizing,
                    'Compra Valorizada': valued_cost if buy == 'Si' and is_obs != 'OB' else "0",
                    'Venta valorizada': int(round(float(price) * avg_sales)),
                    'Valorizado': int(round(float(cost_price) * stock)),
                    'Demora en dias': str(lead_time),
                    'Fecha próx. compra': str(next_buy) if days_of_coverage != 9999 else "---",
                    'Caracterizacion': characterization,
                    'Sobrante (unidades)': overflow_units,
                }

                results.append(stock)

            return results

        except Exception as err:
            print("ERROR CALCULO REAPRO", err)
            traceback.print_exc()

    def post(self, request):
        reapro_params = request.data.get('params', {})
        project = request.data.get('project_name')
        client = request.user.userinformation.get().client 
        type_of_stock = request.data.get('type_of_stock')
        next_buy_days = request.data.get('next_buy_days', 15)
        is_forecast = request.data.get('is_forecast', False) 
        scenario = request.data.get('scenario', None)
        forecast_periods = request.data.get('forecast_periods', None)
        is_drp = request.data.get('is_drp')
        historical_periods = request.data.get("historical_periods", None)
        d = reapro_params.get('d', 1)
        k = reapro_params.get('k', 1)

        products = Product.objects.filter(file__project__name=project, file__project__client=client)
        file = File.objects.filter(project__name=project, project__client=client, file_type='historical').first()

        if historical_periods is not None:

            latest_dates = Sales.objects.filter(
                product__file__project__name=project
            ).values_list('date', flat=True).distinct()

            sorted_dates = sorted(latest_dates, reverse=True)
            latest_dates = sorted_dates[:int(historical_periods)]

            sales_data = Sales.objects.filter(
                product__in=products,
                date__in=latest_dates
            )

        else:
            sales_data = Sales.objects.filter(product__in=products)

        sales_calc = sales_data.values('product_id').annotate(
            avg_row_historical=Avg('sale'),
            desv_historical=StdDev('sale'),
            total_sales_historical=Sum('sale')
        ).annotate(
            coefficient_of_variation_historical=F('desv_historical') / F('avg_row_historical'),
            avg_sales_per_day_historical=F('avg_row_historical') / 30,
            desv_per_day_historical=F('desv_historical') / 30
        )

        stock_data = Stock.objects.filter(product__in=products).values()
        stock_dict = {
            item['product_id']: item
            for item in stock_data
        }

        sales_data = {sale_info['product_id']: sale_info for sale_info in sales_calc}

        data = [
            {
                'product_id': item.id,
                'stock_info': True if item.id in stock_dict else False,
                'sales_info': True if item.id in sales_data else False, 
                'forecast_info': True,
                'family': item.family,
                'region': item.region,
                'salesman': item.salesman,
                'client': item.client,
                'category': item.category,
                'subcategory': item.subcategory,
                'description': item.description,
                'sku': item.sku,
                **sales_data.get(int(item.id), {
                    "avg_row_historical" : 0, 
                    "desv_historical" : 0, 
                    "total_sales_historical" : 0, 
                    "coefficient_of_variation_historical" : 0, 
                    "avg_sales_per_day_historical" : 0, 
                    "desv_per_day_historical" : 0.0, 
                }), 
                'total_sales_forecast': 0,
                'avg_row_forecast': 0,
                'avg_sales_per_day_forecast': 0,
                'desv_per_day_forecast': 0,
                'desv_forecast': 0,
                'coefficient_of_variation_forecast':0,
                **stock_dict.get(int(item.id), {
                    "stock" : 0, 
                    "sales_order_pending_delivery" : 0, 
                    "safety_lead_time" : 0, 
                    "safety_stock" : 0, 
                    "lead_time" : 0, 
                    "cost_price" : 0.0, 
                    "price" : 0.0, 
                    "eoq" : 0, 
                    "service_level" : 0, 
                    "desv_std" : 0, 
                    "purchase_order" : 0,
                    "lot_sizing" : 0,
                    "abc" : "N/A", 
                    "xyz" : "N/A", 
                    "purchase_unit" : 0, 
                    "make_to_order" : "N/A",  
                    "slow_moving" : "N/A", 
                    "drp_lot_sizing" : 0, 
                    "drp_safety_stock" : 0, 
                    "drp_lead_time" : 0, 
                    "supplier_sku_code" : "N/A" 
                }) 
            }
            for item in products
        ]

        if is_forecast and scenario is not None and forecast_periods is not None:
            try:
                scenario = Scenario.objects.get(name=scenario)
                
                predicted_sales_dates = PredictedSale.objects.filter(
                    scenario=scenario, 
                    best_model=True
                ).values_list('date', flat=True).distinct()
                
                future_dates = [date for date in predicted_sales_dates if date > file.project.max_historical_date]

                sorted_dates = sorted(future_dates)
                latest_dates = sorted_dates[:int(forecast_periods)]

                predicted_sales = PredictedSale.objects.filter(
                    scenario=scenario,
                    best_model=True,
                    date__in=latest_dates
                )
                
                predicted_calc = predicted_sales.values('product_id').annotate(
                    avg=Avg('sale'),
                    desv=StdDev('sale'),
                    sum=Sum('sale')
                ).annotate(
                    coeff=F('desv') / F('avg'),
                    avg_day=F('avg') / 30,
                    desv_day=F('desv') / 30
                )

                predicted_calc_dict = {
                    item['product_id']: {
                        'total_sales_forecast': item['sum'] if item['sum'] is not None else 0,
                        'avg_row_forecast':  item['avg'] if item['avg'] is not None else 0,
                        'avg_sales_per_day_forecast': item['avg_day'] if item['avg_day'] is not None else 0,
                        'desv_per_day_forecast': item['desv_day'] if item['desv_day'] is not None else 0,
                        'desv_forecast': item['desv'] if item['desv'] is not None else 0,
                        'coefficient_of_variation_forecast': item['coeff'] if item['coeff'] is not None else 0,
                    }
                    for item in predicted_calc
                }

                data = [
                    {
                        **item,  
                        **predicted_calc_dict.get(item['product_id'], {
                            'forecast_info': False,
                            'total_sales_forecast':0,
                            'avg_row_forecast':  0,
                            'avg_sales_per_day_forecast':0,
                            'desv_per_day_forecast': 0,
                            'desv_forecast': 0,
                            'coefficient_of_variation_forecast': 0,
                        })  
                    }
                    for item in data
                ]
                
            except Scenario.DoesNotExist:
                return Response({"error": "Scenario not found."}, status=status.HTTP_404_NOT_FOUND)

        if type_of_stock == 'reapro':
            reapro = self.calculate_reapro(data=data, d=d, k=k, is_forecast=is_forecast, next_buy_days=next_buy_days)
            traffic_light = self.calculate_traffic_light(reapro)

            if is_drp:
                drp = self.calculate_drp(products=reapro, is_forecast=is_forecast)
                return Response(data={"data": drp, "traffic_light": traffic_light}, status=status.HTTP_200_OK)
            
            else:
                return Response(data={'data': reapro, 'traffic_light': traffic_light}, status=status.HTTP_200_OK)
    
        elif type_of_stock == 'safety_stock':
            safety_stock = self.calculate_safety_stock(data=data)
            return Response(data=safety_stock, status=status.HTTP_200_OK)


class StockModelViewSet(viewsets.ModelViewSet):
    queryset = Stock.objects.all()
    permission_classes = [ IsAuthenticated ]
    serializer_class = StockSerializer

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
            f"product__{field}__in": request.GET.getlist(field)
            for field in filter_fields
            if request.GET.getlist(field)
        }

        return queryset.filter(**filters)

    def list(self, request, *args, **kwargs):
        page_number = request.GET.get('page', '1')
        projectname = request.GET.get('project')
        export = request.GET.get('export', False)
        client = request.user.userinformation.get().client

        query = self.get_queryset().filter(product__file__project__name=projectname, product__file__project__client=client)
        query = self.apply_product_filters(query, request)

        if page_number == 'all':
            serializer = self.get_serializer(query, many=True)
            response_data = {
                "total_items": query.count(),
                "products": serializer.data,
            }

        else:
            paginator = Paginator(query, 10) 
            page = paginator.get_page(int(page_number))
            serializer = self.get_serializer(page.object_list, many=True)

            response_data = {
                "total_items": paginator.count,
                "total_pages": paginator.num_pages,
                "current_page": page_number,
                "products": serializer.data,
            }

        return Response(response_data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['patch'], url_path='bulk-update')
    def bulk_update_safety_stock(self, request):
        bulk_data = request.data

        updates = []
        product_ids = []

        for entry in bulk_data:
            product_id, stock_value = list(entry.items())[0]  
            
            try:
                product_ids.append(int(product_id))  
                stock_value = int(stock_value)  
            except (ValueError, TypeError):
                raise ValidationError(f"Invalid PRODUCT ID or STOCK VALUE: {entry}. Both must be integers.")

        stocks = Stock.objects.filter(product_id__in=product_ids)

        # Preparar las actualizaciones
        for stock in stocks:
            stock_value = next((int(entry[str(stock.product_id)]) for entry in bulk_data if str(stock.product_id) in entry), None)
            if stock_value is not None:
                stock.safety_stock = stock_value
                updates.append(stock)

        # Realizar la actualización masiva
        with transaction.atomic():
            Stock.objects.bulk_update(updates, ['safety_stock'])

        return Response({"detail": "Safety stock updated successfully."}, status=status.HTTP_200_OK)

    

## UPDATE SALES CSV ##
class UploadInventoryCSV(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        data = request.data.get('data')
        project = request.data.get('project')
        client = request.user.userinformation.get().client 
        products_len = len(data)

        if not data:
            return Response({"error": "no_data_provided"}, status=status.HTTP_400_BAD_REQUEST)

        df = pd.DataFrame(data=data)

        no_stock_info = []
        updated_products = []

        for column in ["Family", "Region", "SKU", "Category", "Subcategory", "Client", "Salesman", "Description"]:
            df[column] = df[column].apply(
                lambda x: ''.join(
                    c for c in unicodedata.normalize('NFD', x.strip()) if not unicodedata.combining(c)
                ) if isinstance(x, str) else x
            )

        if 'Product ID' in df.columns and df['Product ID'].notnull().any():
            df.rename(columns={'Product ID': 'product_id'}, inplace=True)
            df.drop(columns=["Family", "Region", "SKU", "Category", "Subcategory", "Client", "Salesman", "Description"], inplace=True)
        
        else:
            df.rename(columns={
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
            
            df['hash'] = df.apply(lambda row: '-'.join(str(row[field]) for field in fields), axis=1)
            
            hash_to_product_id = {
                '-'.join(str(getattr(product, field)) for field in fields): product.id
                for product in Product.objects.filter(file__project__name=project, file__project__client=client)
            }

            df['product_id'] = df['hash'].map(hash_to_product_id)
            df.dropna(subset=['product_id'], inplace=True)
            columns = ['product_id'] + [col for col in df.columns if col != 'product_id']
            df = df[columns]
            df.drop(columns=['hash', "family", "region", "sku", "category", "subcategory", "client", "salesman", "description"], inplace=True)
        
        df.rename(columns={'Stock': 'stock', 'Sales Order Pending Deliverys': 'sales_order_pending_delivery', 'Purchase Order': 'purchase_order'}, inplace=True)

        records = df.to_dict(orient="records")

        existing_records = {
            (stock_info.product_id): stock_info
            for stock_info in Stock.objects.filter(
                product_id__in=df["product_id"].unique()
            )
        }

        for record in records:
            key = (int(record["product_id"]))
            
            if key in existing_records:
                existing_record = existing_records[key]
                existing_record.stock = record['stock']
                existing_record.sales_order_pending_delivery = record['sales_order_pending_delivery']
                existing_record.purchase_order = record['purchase_order']
                updated_products.append(existing_record)

        with transaction.atomic():
            if updated_products:
                Stock.objects.bulk_update(updated_products, ['stock', 'sales_order_pending_delivery', 'purchase_order'], batch_size=10000)
        
        if no_stock_info:
            return Response(data={'message': 'succeed', 'skus_with_no_stock_info': no_stock_info}, status=status.HTTP_200_OK)
        
        elif len(no_stock_info) == products_len:
            return Response(data={'error': 'no_stock_info_found'}, status=status.HTTP_400_BAD_REQUEST)
        
        else:
            return Response(data={'message': 'succeed'}, status=status.HTTP_200_OK)