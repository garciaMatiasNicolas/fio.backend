from django.db import models
#from users.models import User


class Clients(models.Model):
    name = models.CharField(max_length=200, unique=True)
    address = models.CharField(max_length=200, blank=True, null=True)
    state = models.CharField(max_length=200, blank=True, null=True)
    country = models.CharField(max_length=200, blank=True, null=True)
    phone = models.CharField(max_length=200, blank=True, null=True)
    category = models.CharField(max_length=200, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=False, blank=True, null=True)
    logo = models.ImageField(upload_to='logos/', blank=True, null=True)


class BusinessRules(models.Model):
    client = models.ForeignKey(Clients, on_delete=models.CASCADE, null=False, blank=False)
    status = models.BooleanField(default=True, null=False, blank=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now_add=False, blank=True, null=True)
    fields = models.JSONField(null=False, blank=False, default=list)
