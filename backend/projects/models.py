from django.db import models
from users.models import User
from clients.models import Clients


class Projects(models.Model):
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    updated_at = models.DateTimeField(auto_now_add=False, null=True, blank=True)
    name = models.CharField(max_length=100, unique=True)
    status = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    client = models.ForeignKey(Clients, on_delete=models.CASCADE)
    max_historical_date = models.DateField(blank=True, null=True)
    periods_per_year = models.IntegerField(default=12)