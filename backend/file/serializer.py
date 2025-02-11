from rest_framework import serializers
from .models import File, Product, Sales, ExogenousVariables, ProjectedExogenousVariables
from clients.models import BusinessRules
import pandas as pd
from projects.models import Projects 
from django.shortcuts import get_object_or_404
from projects.models import Projects
from django.db.models import Max
import unicodedata
from django.db import transaction
import os
import csv
from django.db import connection, transaction
from django.conf import settings


class FileSerializer(serializers.ModelSerializer):
    data = serializers.ListField(child=serializers.DictField(), write_only=True)
    file_type = serializers.CharField(required=True)
    project_name = serializers.CharField(required=True) 

    class Meta:
        model = File
        exclude = ('uploaded_at', 'uploaded_by', 'project')

    def format_dates(self, df: pd.DataFrame, file_type: str):
        if file_type == 'historical':
            date_cols = df.columns[10:]
        elif file_type == 'exogenous' or file_type=='launch_data':
            date_cols = df.columns[9:]
        else:
            raise ValueError("File type not supported")

        date_cols_format = [pd.to_datetime(col, format='%d/%m/%Y', errors='coerce').strftime('%Y-%m-%d') for col in date_cols]
        map_cols = dict(zip(date_cols, date_cols_format))
     
        df.rename(columns=map_cols, inplace=True)
        
        return df

    def validate(self, attrs):
        file_type = attrs.get('file_type')
        request = self.context.get('request')
        client = request.user.userinformation.get().client

        data = attrs.get('data')

        # Definir las columnas requeridas según el tipo de archivo
        required_columns = {
            'historical': {"Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description", "Periods Per Year"},
            'stock': {"Stock", "Sales Order Pending Deliverys", "Safety Lead Time (days)", 
                "Safety stock (days)", "Lead Time", "Cost Price", "Price", "EOQ (Economical order quantity)", 
                "Service Level", "Desv Est Lt Days", "Purchase Order", "Lot Sizing", 
                "ABC", "XYZ", "Purchase unit", "Make to order", "Slow moving", 
                "DRP Lot sizing", "DRP safety stock (days)", "DRP lead time", "Supplier SKU code"
            },
            'exogenous': {"Variable","Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"},
            'projected_exogenous': {"Variable","Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"},
            'discontinued_data': {'Product ID', 'Discontinued'},
            'replace_data': {'Product ID', 'Replace With (SKU)'},
            'launch_data': {"Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"}
        }

        if file_type not in required_columns:
            raise serializers.ValidationError({"Tipo de archivo no aceptado"})

        missing_columns = required_columns[file_type] - set(data[0].keys())
        if missing_columns:
            raise serializers.ValidationError(f"Faltan las siguientes columnas: {missing_columns}")

        # Validar que el proyecto existe
        project_name = attrs.get('project_name')
        project = get_object_or_404(Projects, client=client, name=project_name)
        attrs['project'] = project

        if file_type == 'historical':
            df = pd.DataFrame(data).dropna(how='all')
            historical_dates = df.columns[10:]

            if len(historical_dates) < 8:
                raise serializers.ValidationError("El archivo 'historical' debe contener al menos 8 periodos de fechas de venta histórica. (Recomendamos 24)")

        return attrs

    def create(self, validated_data):
        request = self.context.get('request')
        user = request.user
        project = validated_data.get('project')
        file_type = validated_data.get('file_type')
        data = validated_data.pop('data')

        try:
            File.objects.filter(project=project,  file_type=file_type).first().delete()
        
        except:
            pass

        df = pd.DataFrame(data)
        
        client_name = project.client.name
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads", client_name)
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        file_name = f"{file_type}_{project.name}_{user.username}.csv"
        file_path = os.path.join(upload_dir, file_name)

        df.to_csv(file_path, index=False)
        file_instance = File.objects.create(
            project=project,
            uploaded_by=user,
            file_type=file_type,
            file_path=file_path
        )

        try:
            self.process_data(data, file_instance, project, file_type)

        except Exception as e:
            print(str(e))
            file_instance.delete()  
            raise serializers.ValidationError(f"Error processing data: {str(e)}")

        return file_instance
    
    def process_data(self, data, file_instance: File, project: Projects, file_type: str):
        df = pd.DataFrame(data).dropna(how='all')

        if file_type == 'historical':
            df = df.reset_index() 
            df = df.rename(columns={'index': 'ID'}) 
            df = self.format_dates(df=df, file_type='historical')
            project.max_historical_date = df.columns[-1]
            project.periods_per_year = df['Periods Per Year'].iloc[0]
            project.save()
            df = df.drop(columns=['Periods Per Year'])
            self.process_products(file_instance=file_instance, df=df)
            self.process_sales(df=df, project=project)
        
        if file_type == 'launch_data':
            df = df.reset_index() 
            df = df.rename(columns={'index': 'ID'}) 
            df = self.format_dates(df=df, file_type='launch_data')

            self.process_products(file_instance=file_instance, df=df)
            self.process_sales(df=df, project=project)

        if file_type == 'stock':
            self.process_stock(project=project, df=df, file_instance=file_instance)
        
        if file_type == 'exogenous':
            df = self.format_dates(df=df, file_type='exogenous')
            self.process_exog_data(df=df, file_instance=file_instance, project=project)
        
        if file_type == 'projected_exogenous':
            df = self.format_dates(df=df, file_type='exogenous')
            self.process_exog_data_projected(df=df, file_instance=file_instance, project=project)
        
        if file_type == 'discontinued_data':
            self.process_discontinued_data(df=df)
        
        if file_type == 'replace_data':
            self.process_replace_data(df=df)
    
    @staticmethod
    def kpis(df: pd.DataFrame):
        df_sale_data = df.iloc[:, 9:].apply(pd.to_numeric, errors='coerce').fillna(0)

        # Validación mínima de columnas
        if df_sale_data.shape[1] < 8:
            df["ytd"] = 0
            df["qtd"] = 0
            df["mtd"] = 0
            return df

        ytd = []
        qtd = []
        mtd = []

        for _, row in df_sale_data.iterrows():
            total_columns = row.shape[0]

            # Cálculo de QTD (Últimas 4 vs las 4 anteriores)
            qtd_current = row.iloc[-4:].sum()  # Suma de las últimas 4 columnas
            qtd_previous = row.iloc[-8:-4].sum()  # Suma de las 4 columnas anteriores
            if qtd_previous > 0:
                qtd_growth = ((qtd_current - qtd_previous) / qtd_previous) * 100
            else:
                qtd_growth = 0
            qtd.append(qtd_growth)

            # Cálculo de MTD (Última columna vs Penúltima columna)
            mtd_current = row.iloc[-1]  # Última columna
            mtd_previous = row.iloc[-2]  # Penúltima columna
            if mtd_previous > 0:
                mtd_growth = ((mtd_current - mtd_previous) / mtd_previous) * 100
            else:
                mtd_growth = 0
            mtd.append(mtd_growth)

            if total_columns >= 24:
                ytd_second_half = row.iloc[-12:].sum()  
                ytd_first_half = row.iloc[-24:-12].sum()  
            elif total_columns >= 12:
                mid_point = total_columns // 2  # División en dos mitades iguales o cercanas
                ytd_first_half = row.iloc[:mid_point].sum()  # Suma de la primera mitad
                ytd_second_half = row.iloc[mid_point:].sum()  # Suma de la segunda mitad
            else:
                ytd_first_half = 0
                ytd_second_half = 0

            if ytd_first_half > 0:
                ytd_growth = ((ytd_second_half - ytd_first_half) / ytd_first_half) * 100
            else:
                ytd_growth = 0
            ytd.append(ytd_growth)

        # Cálculo del promedio de las últimas 12 columnas
        df['avg'] = df_sale_data.iloc[:, -12:].mean(axis=1)
        df["ytd"] = ytd
        df["qtd"] = qtd
        df["mtd"] = mtd

        return df

    def process_products(self, file_instance: File, df: pd.DataFrame):
        df_with_kpis = self.kpis(df=df)
        df = df.dropna(how='all')

        df_products = df.iloc[:, :9].fillna("")
        
        for column in ['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU', 'Description']:
            df_products[column] = df_products[column].apply(
                lambda x: ''.join(
                    c for c in unicodedata.normalize('NFD', x.strip()) if not unicodedata.combining(c)
                ) if isinstance(x, str) else x
            )

        df_products["ytd"] = df_with_kpis["ytd"]
        df_products["qtd"] = df_with_kpis["qtd"]
        df_products["mtd"] = df_with_kpis["mtd"]
        df_products['avg'] = df_with_kpis['avg'] 

        products_data = df_products.to_dict(orient='records')
        
        def generate_products(file_instance: File, products_data: list):
            for data in products_data:
                yield Product(
                    file=file_instance,
                    template_id=data["ID"],
                    family=data['Family'],
                    region=data['Region'],
                    salesman=data['Salesman'],
                    client=data['Client'],
                    category=data['Category'],
                    subcategory=data['Subcategory'],
                    sku=data['SKU'],
                    description=data['Description'],
                    ytd=data.get("ytd", 0) if pd.notnull(data.get("ytd")) else 0,
                    qtd=data.get("qtd", 0) if pd.notnull(data.get("qtd")) else 0,
                    mtd=data.get("mtd", 0) if pd.notnull(data.get("mtd")) else 0,
                    avg=data.get("avg", 0.0)
                )

        products = list(generate_products(file_instance, products_data))

        with transaction.atomic():
            created_products = Product.objects.bulk_create(products, batch_size=20000)
        
        return created_products 

    def process_sales(self, df: pd.DataFrame, project: Projects):
        # Drop unused columns
        df = df.drop(columns=['ytd', 'qtd', 'mtd', 'avg'])

        df_sales = df.melt(
            id_vars=df.columns[:9],
            value_vars=df.columns[9:], 
            var_name='DATES', 
            value_name='SALES'
        )
        
        # Filter and clean data
        df_sales = df_sales[['ID', 'DATES', 'SALES']]
        df_sales['SALES'] = df_sales['SALES'].fillna(0.0)

        # Map products
        product_ids = df_sales['ID'].unique()
        products = Product.objects.filter(file__project=project, template_id__in=product_ids)
        product_dict = {product.template_id: product.id for product in products}
        df_sales['product_id'] = df_sales['ID'].map(product_dict)

        # Drop rows with unmapped products
        df_sales = df_sales.dropna(subset=['product_id'])
        df_sales = df_sales[['product_id', 'DATES', 'SALES']]

        file_path = os.path.join(settings.LOAD_DATA_INFILE_DIR, 'sales_data.csv')

        # Write DataFrame to the specified CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as temp_file:
            csv_writer = csv.writer(temp_file)
            # Write the CSV header
            csv_writer.writerow(['product_id', 'date', 'sale'])
            # Write the data rows
            csv_writer.writerows(df_sales.values)

        temp_file_name = file_path.replace("\\", "/")
 
        sql = f"""
            LOAD DATA INFILE '{temp_file_name}'
            INTO TABLE file_sales
            FIELDS TERMINATED BY ',' 
            LINES TERMINATED BY '\n'
            IGNORE 1 ROWS
            (product_id, date, sale)
        """
        
        try:
          with connection.cursor() as cursor:
            cursor.execute(sql=sql)  
        except Exception as e:
          print(f"Error inserting sales data: {e}")

    def process_exog_data(self, file_instance: File, df: pd.DataFrame, project: Projects):
        df_exog = df.melt(
            id_vars=["Variable", "Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"],
            var_name="Date",
            value_name="Exog"
        )

        df_exog = df_exog.dropna(subset=["Exog", "Variable"])
        df = df.dropna(axis=1, how="all")
        columns_to_match = [col for col in df_exog.columns if col not in ["Variable", "Date", "Exog"]]

        exog_entries = []

        for _, row in df_exog.iterrows():
            filters = {f"{col.lower()}__iexact": row[col] for col in columns_to_match if pd.notna(row[col])}
            matching_products = Product.objects.filter(file__project=project, **filters).only('id')

            sales = Sales.objects.filter(product__id__in=matching_products, date=row['Date'])

            for sale in sales:
                exogenous = ExogenousVariables(
                    sale=sale,
                    file=file_instance,
                    variable=row["Variable"],
                    exog=row["Exog"]
                )
                exog_entries.append(exogenous)

        if len(exog_entries) > 0:
            ExogenousVariables.objects.bulk_create(exog_entries, batch_size=10000)
        else:
            raise ValueError("No se encontraron productos para las exogenas provistas. Revise las columnas")
    
    def process_exog_data_projected(self, file_instance: File, df: pd.DataFrame, project: Projects):
        df_exog = df.melt(
            id_vars=["Variable", "Family", "Region", "Salesman", "Client", "Category", "Subcategory", "SKU", "Description"],
            var_name="Date",
            value_name="Exog"
        )

        df_exog = df_exog.dropna(subset=["Exog", "Variable"])
        df = df.dropna(axis=1, how="all")
        columns_to_match = [col for col in df_exog.columns if col not in ["Variable", "Date", "Exog"]]

        exog_entries = []
        for _, row in df_exog.iterrows():
            filters = {f"{col.lower()}__iexact": row[col] for col in columns_to_match if pd.notna(row[col])}
            matching_products = Product.objects.filter(file__project=project, **filters).only('id')

            for product in matching_products:
                projected_exog = ProjectedExogenousVariables(
                    product=product,  
                    file=file_instance,
                    variable=row["Variable"],
                    date=row["Date"], 
                    exog=row["Exog"]
                )
                exog_entries.append(projected_exog)

        if len(exog_entries) > 0:
            ProjectedExogenousVariables.objects.bulk_create(exog_entries, batch_size=10000)
        else:
            raise ValueError("No se encontraron productos para las exogenas provistas. Revise las columnas")

    def process_stock(self, project: Projects, df: pd.DataFrame, file_instance: File):
        # Rellenar valores faltantes en el DataFrame
        df = df.fillna({
            'Family': '',
            'Region': '',
            'Salesman': '', 
            'Client': '', 
            'Category': '', 
            'Subcategory': '', 
            'SKU': '', 
            'Description': '',
            'Stock': 0,
            'Supplier SKU code': '',
            'Sales Order Pending Deliverys': 0,
            'Safety Lead Time (days)': 0,
            'Safety stock (days)': 0,
            'Lead Time': 0,
            'Cost Price': 0.0,
            'Price': 0.0,
            'EOQ (Economical order quantity)': 0,
            'Service Level': 0,
            'Desv Est Lt Days': 0,
            'Purchase Order': 0,
            'Lot Sizing': 0,
            'ABC': '',
            'XYZ': '',
            'Purchase unit': 0,
            'Make to order': 0,
            'Slow moving': 0,
            'DRP Lot sizing': 0,
            'DRP safety stock (days)': 0,
            'DRP lead time': 0
        })
        
        df['Make to order'] = df['Make to order'].apply(lambda x: 1 if x == 'MTO' else x)
        df['Slow moving'] = df['Slow moving'].apply(lambda x: 1 if x == 'OB' else x)

        # Diccionario de mapeo: columnas actuales -> campos del modelo
        rename_columns = {
            'Family': 'family',
            'Region': 'region',
            'Salesman': 'salesman', 
            'Client': 'client', 
            'Category': 'category', 
            'Subcategory': 'subcategory', 
            'SKU': 'sku', 
            'Description': 'description',
            'Stock': 'stock',
            'Sales Order Pending Deliverys': 'sales_order_pending_delivery',
            'Safety Lead Time (days)': 'safety_lead_time',
            'Safety stock (days)': 'safety_stock',
            'Lead Time': 'lead_time',
            'Cost Price': 'cost_price',
            'Price': 'price',
            'EOQ (Economical order quantity)': 'eoq',
            'Service Level': 'service_level',
            'Desv Est Lt Days': 'desv_std',
            'Purchase Order': 'purchase_order',
            'Lot Sizing': 'lot_sizing',
            'ABC': 'abc',
            'XYZ': 'xyz',
            'Purchase unit': 'purchase_unit',
            'Make to order': 'make_to_order',
            'Slow moving': 'slow_moving',
            'DRP Lot sizing': 'drp_lot_sizing',
            'DRP safety stock (days)': 'drp_safety_stock',
            'DRP lead time': 'drp_lead_time',
            'Supplier SKU code': 'supplier_sku_code'
        }

        # Renombrar las columnas del DataFrame
        df.rename(columns=rename_columns, inplace=True)

        for column in ['family', 'region', 'salesman', 'client', 'category', 'subcategory', 'sku', 'description']:
            df[column] = df[column].apply(
                lambda x: ''.join(
                    c for c in unicodedata.normalize('NFD', x.strip()) if not unicodedata.combining(c)
                ) if isinstance(x, str) else x
            )

        client = project.client
        business_rule = BusinessRules.objects.filter(client=client).first()
        fields = business_rule.fields

        df['hash'] = df.apply(lambda row: '-'.join(str(row[field]) for field in fields), axis=1)

        # Crear hash-to-product_id dinámicamente usando los mismos campos
        hash_to_product_id = {
            '-'.join(str(getattr(product, field)) for field in fields): product.id
            for product in Product.objects.filter(file__project=project, file__project__client=client)
        }

        df['product_id'] = df['hash'].map(hash_to_product_id)
        columns = ['product_id'] + [col for col in df.columns if col != 'product_id']
        df = df[columns]
        df.drop(columns=['hash'], inplace=True)

        existing_products_data = df[df['product_id'].notnull()].drop(columns=['supplier_sku_code', 'family', 'region', 'salesman', 'client', 'category', 'subcategory', 'sku', 'description'])
        
        new_products_data = df[df['product_id'].isnull()]
        new_products_data.drop(columns=['product_id'], inplace=True)

        max_id = Product.objects.filter(file__project=project).aggregate(Max('template_id'))['template_id__max'] or 0        
        
        ## Create csv ##
        file_path = os.path.join(settings.LOAD_DATA_INFILE_DIR, 'stock_data.csv')

        existing_products_data['file_id'] = file_instance.id
        
        # Write DataFrame to the specified CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as temp_file:
            csv_writer = csv.writer(temp_file)
            # Write the CSV header
            csv_writer.writerow(['product_id', 'stock', 'sales_order_pending_delivery',
                'safety_lead_time', 'safety_stock', 'lead_time', 'cost_price', 'price',
                'eoq', 'service_level', 'desv_std',
                'purchase_order', 'lot_sizing', 'abc', 'xyz', 'purchase_unit',
                'make_to_order', 'slow_moving', 'drp_lot_sizing', 'drp_safety_stock',
                'drp_lead_time', 'file_id'])
            
            # Write the data rows
            csv_writer.writerows(existing_products_data.values)
        
        temp_file_name = file_path.replace("\\", "/")
 
        sql = f"""
            LOAD DATA INFILE '{temp_file_name}'
            INTO TABLE inventory_stock
            FIELDS TERMINATED BY ',' 
            LINES TERMINATED BY '\n'
            IGNORE 1 ROWS
            (product_id, stock, sales_order_pending_delivery,
            safety_lead_time, safety_stock, lead_time, cost_price, price,
            eoq, service_level, desv_std,
            purchase_order, lot_sizing, abc, xyz, purchase_unit,
            make_to_order, slow_moving, drp_lot_sizing, drp_safety_stock,
            drp_lead_time, file_id);
        """

        with connection.cursor() as cursor:
            cursor.execute(sql=sql)
        
        os.remove(file_path)

        ## If there are new products, create them and then assign it their stock information ##
        if not new_products_data.empty:
            products_data = new_products_data.iloc[:, :8].to_dict(orient='records')
            stock_data = new_products_data.iloc[:, 8:].copy()
            stock_data.drop(columns=['supplier_sku_code'], inplace=True)
            

            def generate_products(max_id: int, file_instance: File, products_data: list):
                for data in products_data:
                    max_id += 1
                    yield Product(
                        file=file_instance,
                        template_id=max_id,
                        family=data['family'],
                        region=data['region'],
                        salesman=data['salesman'],
                        client=data['client'],
                        category=data['category'],
                        subcategory=data['subcategory'],
                        sku=data['sku'],
                        description=data['description'],
                        ytd=0,
                        qtd= 0,
                        mtd=0,
                        avg=0.0
                    )

            products = list(generate_products(max_id=max_id, file_instance=file_instance, products_data=products_data))

            with transaction.atomic():
                Product.objects.bulk_create(products, batch_size=30000)
                
                new_product_ids = Product.objects.filter(
                    file=file_instance,
                    template_id__gte=max_id + 1
                ).order_by('template_id').values_list('id', flat=True)

                stock_data['product_id'] = new_product_ids
                stock_data['file_id'] = file_instance.id

                ## Create csv ##
                file_path = os.path.join(settings.LOAD_DATA_INFILE_DIR, 'stock_data_new_products.csv')
        
                # Write DataFrame to the specified CSV file
                with open(file_path, mode='w', newline='', encoding='utf-8') as temp_file:
                    csv_writer = csv.writer(temp_file)
                    # Write the CSV header
                    csv_writer.writerow(['stock', 'sales_order_pending_delivery',
                        'safety_lead_time', 'safety_stock', 'lead_time', 'cost_price', 'price',
                        'eoq', 'service_level', 'desv_std',
                        'purchase_order', 'lot_sizing', 'abc', 'xyz', 'purchase_unit',
                        'make_to_order', 'slow_moving', 'drp_lot_sizing', 'drp_safety_stock',
                        'drp_lead_time', 'product_id', 'file_id'])
                    
                    # Write the data rows
                    csv_writer.writerows(stock_data.values)
                
                temp_file_name = file_path.replace("\\", "/")
        
                sql = f"""
                    LOAD DATA INFILE '{temp_file_name}'
                    INTO TABLE inventory_stock
                    FIELDS TERMINATED BY ',' 
                    LINES TERMINATED BY '\n'
                    IGNORE 1 ROWS
                    (stock, sales_order_pending_delivery,
                    safety_lead_time, safety_stock, lead_time, cost_price, price,
                    eoq, service_level, desv_std,
                    purchase_order, lot_sizing, abc, xyz, purchase_unit,
                    make_to_order, slow_moving, drp_lot_sizing, drp_safety_stock,
                    drp_lead_time, product_id, file_id);
                """

                with connection.cursor() as cursor:
                    cursor.execute(sql=sql)
                
                os.remove(file_path)

    def process_discontinued_data(self, df: pd.DataFrame):
        filtered_df = df[(df['Product ID'].notnull()) & (df['Discontinued'] == 1)]
        product_ids = filtered_df['Product ID'].tolist()
        products = Product.objects.filter(id__in=product_ids)
        products.update(discontinued=True)
    
    def process_replace_data(self, df: pd.DataFrame):
        filtered_df = df[(df['Product ID'].notnull()) & (df['Replace With (SKU)'].notnull())]
        filtered_df.drop(columns=['Family', 'Region', 'Salesman', 'Client', 'Category', 'Subcategory', 'SKU', 'Description'], inplace=True)

        for _, row in filtered_df.iterrows():
            
            product_to_replace = Sales.objects.filter(product__id=row['Product ID'])
            replace_with = Sales.objects.filter(product__sku=row['Replace With (SKU)'])
           
            if len(product_to_replace) == 0 or len(replace_with) == 0:
                raise ValueError( "No se encontraron ventas para los productos especificados")

            sales_to_add = Sales.objects.filter(product__id=row['Product ID'])
            for sale in sales_to_add:
                existing_sale = Sales.objects.filter(
                    product__sku=row['Replace With (SKU)'], date=sale.date
                ).first()

                existing_sale.sale += sale.sale
                existing_sale.save()
            
            product_to_replace.delete()
            Product.objects.get(id=row['Product ID']).delete()

    def to_representation(self, instance):
        return {
            'id': instance.id,
            'project': instance.project.name,
            'uploaded_by': f'{instance.uploaded_by.first_name} {instance.uploaded_by.last_name}',
            'uploaded_at': instance.uploaded_at,
            'file_type': instance.file_type
        }


class ProductSerializer(serializers.ModelSerializer):
    project_name = serializers.CharField(write_only=True)
    group = serializers.CharField(write_only=True, required=False) 

    class Meta:
        model = Product
        exclude = ('file', 'template_id')

    def validate(self, data):
        project_name = data.get('project_name')
        try:
            Projects.objects.get(name=project_name)
        except Projects.DoesNotExist:
            raise serializers.ValidationError({"project_name": "Project does not exist."})

        file = File.objects.filter(project__name=project_name, file_type='historical').first()
        if not file:
            raise serializers.ValidationError({"file": "No historical file found for this project."})
    
        template_id = Product.objects.filter(file__project__name=project_name).aggregate(Max('template_id'))['template_id__max']

        data['template_id'] = template_id
        data['file'] = file

        return data

    def create(self, validated_data):
        group = validated_data.pop('group', None)
        project = validated_data.pop('project_name', None)

        if group is not None:
            group_values = Product.objects.filter(file__project__name=project).values_list(group, flat=True).distinct()

            if len(group_values) > 1:  
                prod_entries = []

                for value in group_values:
                    temp_id = validated_data.get('template_id')  

                    product_data = validated_data.copy()
                    product_data[group] = value 
                    product_data["template_id"] = temp_id

                    prod_entries.append(Product(**product_data))

                    temp_id += 1

                Product.objects.bulk_create(prod_entries, batch_size=1000)
                return prod_entries[0]

            else:
                return super().create(validated_data)

        else:
            return super().create(validated_data)

    def update(self, instance, validated_data):
        validated_data.pop('project_name', None)
        return super().update(instance, validated_data)

    def to_representation(self, instance):
        return {
            "ID": instance.id,
            "Familia": instance.family,
            "Region": instance.region,
            "Vendedor": instance.salesman,
            "Cliente": instance.client,
            "Categoria": instance.category,
            "Subcategoria": instance.subcategory,
            "SKU":instance.sku,
            "Descripcion": instance.description,
            "YTD": instance.ytd,
            "QTD":instance.qtd,
            "MTD":instance.mtd,
            "Promedio (Ult. 12)": instance.avg,
            "Estado": "Descontinuado" if instance.discontinued else "Activo"
        }


class SalesSerializer(serializers.ModelSerializer):

    class Meta:
        model = Sales
        fields = '__all__'
