from django.db import models
from django.contrib.auth.models import User, AbstractUser
from django.conf import settings
from clients.models import Clients


class User(AbstractUser):
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=False)


class UserInformation(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='userinformation')
    updated_at = models.DateTimeField(auto_now_add=False, blank=True, null=True)
    position = models.CharField(max_length=250, blank=True, null=True, default='Administrador')
    phone = models.CharField(max_length=250, blank=True, null=True)
    address = models.CharField(max_length=250, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    profile_pic = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    birth_date = models.DateField(blank=True, null=True)
    client = models.ForeignKey(Clients, on_delete=models.CASCADE, related_name='client')
    # preferences = models.JSONField(default=list)
