from .views import ReaproAPIView, StockModelViewSet, UploadInventoryCSV
from django.urls import path
from rest_framework import routers

router_stock = routers.DefaultRouter()
router_stock.register('inventory/stock', StockModelViewSet, basename='stock')

reapro = ReaproAPIView.as_view()
update_inventory = UploadInventoryCSV.as_view()

urlpatterns = router_stock.urls + [
    path('inventory/reapro/', reapro, name='reapro'),
    path('inventory/update/', update_inventory, name='update_inventory')
]
