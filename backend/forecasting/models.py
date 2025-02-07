from django.db import models
from users.models import User
from projects.models import Projects
from file.models import Product, File
from clients.models import Clients


class Scenario(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    runned_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='user')
    project = models.ForeignKey(Projects, on_delete=models.CASCADE, related_name='project')
    historical_file = models.ForeignKey(File, on_delete=models.CASCADE, related_name='historical_file')
    client = models.ForeignKey(Clients, on_delete=models.CASCADE, related_name='client_scenario')
    name = models.CharField(max_length=200)
    pred_p = models.IntegerField()
    test_p = models.IntegerField()
    error_p = models.IntegerField(default=0)
    replace_negatives = models.BooleanField(default=False)
    detect_outliers = models.BooleanField(default=False)
    seasonal_periods = models.IntegerField(blank=True, null=True)
    error_type = models.CharField(max_length=250)
    is_daily = models.BooleanField(default=False, blank=True, null=True)
    explosive = models.FloatField(default=0.0, blank=True, null=True)
    filter_products = models.BooleanField(default=True)
    additional_params = models.JSONField(default=dict)
    models = models.JSONField(default=list)
    

class PredictedSale(models.Model):
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, related_name='scenario')
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=False, related_name='product_predicted')
    sale = models.FloatField(null=False, blank=False)
    model = models.CharField(max_length=200)
    date = models.DateField()
    colaborated_sale = models.FloatField(default=0.0, null=True, blank=True)
    best_model = models.BooleanField(default=False, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now_add=False, blank=True, null=True)
    updated_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)


class MetricsScenarios(models.Model):
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, null=False, blank=False)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=False, blank=False)
    best_model = models.BooleanField(default=False, null=True, blank=True)
    error = models.FloatField()
    last_period_error = models.FloatField()
    model = models.CharField(max_length=200)
    ytg = models.IntegerField(null=True, blank=True)
    qtg = models.IntegerField(null=True, blank=True)
    mtg = models.IntegerField(null=True, blank=True)
    cluster = models.CharField(max_length=250, null=True, blank=True)
    abc = models.CharField(max_length=250, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now_add=False, blank=True, null=True)
