from rest_framework import serializers
from .models import Stock


class StockSerializer(serializers.ModelSerializer):

    class Meta:
        model = Stock
        fields = '__all__'
    
    def to_representation(self, instance):
        product_data = {
            'ID': instance.product.id,
            'Familia': instance.product.family,
            'Region': instance.product.region,
            'Vendedor': instance.product.salesman,
            'Cliente': instance.product.client,
            'Categoria': instance.product.category,
            'Subcategoria': instance.product.subcategory,
            'Descripcion': instance.product.description,
            'SKU': instance.product.sku,
            'Estado': 'Descontinuado' if instance.product.discontinued else 'Activo',
        }

        stock_data = {
            'Stock Físico': instance.stock,
            'Ordenes de venta pendientes': instance.sales_order_pending_delivery,
            'Tiempo de demora (seguridad)': instance.safety_lead_time,
            'Stock de seguridad': instance.safety_stock,
            'Tiempo de demora (Proveedor)': instance.lead_time,
            'Precio producto': instance.cost_price,
            'Precio': instance.price,
            'Lote optimo (Calculado)': instance.eoq,
            'Nivel de servicio': instance.service_level,
            'Días demorados de los pedidos tradicionales': instance.desv_std,
            'Ordenes de compra pendientes': instance.purchase_order,
            'Lote de compra': instance.lot_sizing,
            'ABC': instance.abc,
            'XYZ': instance.xyz,
            'Unidad de compra': instance.purchase_unit,
            'Make to Order': instance.make_to_order,
            'Slow Moving': instance.slow_moving,
            'Lote de compra DRP': instance.drp_lot_sizing,
            'Stock de seguridad DRP': instance.drp_safety_stock,
            'Tiempo de demora DRP': instance.drp_lead_time,
            'SKU del Proveedor': instance.supplier_sku_code,
        }

        return {**product_data, **stock_data}