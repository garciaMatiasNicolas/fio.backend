from .views import FileViewSet, HistoricalSalesAPIView, ProductViewSet, UploadSalesCSV, SalesViewSet, ProductFiltersAPIView, GraphicOutliersAPIView, ExogenousVariablesViews
from rest_framework.routers import DefaultRouter
from .export import ExportToCsvAPIView, DownloadFileAPIView
from django.urls import path

router_file = DefaultRouter()
router_product = DefaultRouter()
router_sales = DefaultRouter()

router_file.register('file', FileViewSet, basename='file_routes')
router_product.register('product', ProductViewSet, basename='product_routes')
router_sales.register('sales', SalesViewSet, basename='sales_routes')

urlpatterns = router_file.urls + router_product.urls + router_sales.urls + [
    path('historical/exploration/sales/', HistoricalSalesAPIView.as_view(), name='historical_sales_view'),
    path('products/filters/', ProductFiltersAPIView.as_view(), name='filters'),
    path('sale/update-create/', UploadSalesCSV.as_view(), name='sales_csv'),
    path('historical/exploration/outliers/', GraphicOutliersAPIView.as_view(), name='outliers_graphic_view'),
    path('historical/exploration/exogenous/graphic/', ExogenousVariablesViews.ExogenousVariablesGraphicAPIView.as_view(), name='exogenous_graphic_view'),
    path('historical/exploration/exogenous/matrix/', ExogenousVariablesViews.AllocationMatrixAPIView.as_view(), name='allocation_matrix_view'),
    path('file/export/', ExportToCsvAPIView.as_view(), name='export_to_csv'),
    path('file/download/<int:file_id>/', DownloadFileAPIView.as_view(), name='download_uploads')
]

