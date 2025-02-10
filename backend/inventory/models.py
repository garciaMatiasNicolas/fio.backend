from django.db import models
from file.models import Product, File


class Stock(models.Model):
    file = models.ForeignKey(File, related_name='file', null=True, blank=True, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, related_name='stock', on_delete=models.CASCADE)
    stock = models.IntegerField(null=True, blank=True)
    sales_order_pending_delivery = models.IntegerField(null=True, blank=True)
    safety_lead_time = models.IntegerField(null=True, blank=True)
    safety_stock = models.IntegerField(null=True, blank=True)
    lead_time = models.IntegerField(null=True, blank=True)
    cost_price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    price = models.DecimalField(max_digits=20, decimal_places=2, null=True, blank=True)
    eoq = models.IntegerField(null=True, blank=True)
    service_level = models.IntegerField(null=True, blank=True)
    desv_std = models.IntegerField(null=True, blank=True)
    purchase_order = models.IntegerField(null=True, blank=True)
    lot_sizing = models.IntegerField(null=True, blank=True)
    abc = models.CharField(max_length=5, null=True, blank=True)
    xyz = models.CharField(max_length=5, null=True, blank=True)
    purchase_unit = models.IntegerField(null=True, blank=True)
    make_to_order = models.BooleanField(default=False, null=True, blank=True) 
    slow_moving = models.BooleanField(default=False, null=True, blank=True)
    drp_lot_sizing = models.IntegerField(null=True, blank=True)
    drp_safety_stock = models.IntegerField(null=True, blank=True)
    drp_lead_time = models.IntegerField(null=True, blank=True)
    supplier_sku_code = models.CharField(max_length=300,null=True, blank=True)