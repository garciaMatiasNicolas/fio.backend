from users.models import User
from django.db import models
from projects.models import Projects
import os


class File(models.Model):
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=False, related_name='uploaded_by')
    project = models.ForeignKey(Projects, on_delete=models.CASCADE, null=False, related_name="project_related")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_type = models.CharField(max_length=250, default="historical")
    file_path = models.CharField(max_length=450, blank=True, null=True)

class Product(models.Model):
    file = models.ForeignKey(File, on_delete=models.CASCADE, null=False, related_name='file_related')
    template_id = models.IntegerField(db_index=True)
    family = models.CharField(max_length=255, null=True, blank=True)
    region = models.CharField(max_length=255, null=True, blank=True)
    salesman = models.CharField(max_length=255, null=True, blank=True)
    client = models.CharField(max_length=255, null=True, blank=True)
    category = models.CharField(max_length=255, null=True, blank=True)
    subcategory = models.CharField(max_length=255, null=True, blank=True)
    description = models.CharField(max_length=1000, null=True, blank=True)
    sku = models.CharField(max_length=255, null=True, blank=True)
    ytd = models.IntegerField(null=True, blank=True)
    qtd = models.IntegerField(null=True, blank=True)
    mtd = models.IntegerField(null=True, blank=True)
    avg = models.FloatField(default=0.0, null=True, blank=True)
    discontinued = models.BooleanField(default=False, null=False, blank=False)


class Sales(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=False, related_name='product')
    date = models.DateField(null=False, blank=False)
    sale = models.FloatField(null=False, blank=False)


class ExogenousVariables(models.Model):
    sale = models.ForeignKey(Sales, on_delete=models.CASCADE, related_name='sale_related', default=None)
    file = models.ForeignKey(File, on_delete=models.CASCADE, null=False)
    variable = models.CharField(max_length=250)
    exog = models.FloatField(null=False, blank=False, default=0.0)


class ProjectedExogenousVariables(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='prod_related', default=None)
    file = models.ForeignKey(File, on_delete=models.CASCADE, null=False)
    variable = models.CharField(max_length=250)
    date = models.DateField(null=False, blank=False)
    exog = models.FloatField(null=False, blank=False, default=0.0)
