from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('users.urls')),
    path('api/', include('file.urls')),
    path('api/', include('clients.urls')),
    path('api/', include('projects.urls')),
    path('api/', include('projects.urls')),
    path('api/', include('inventory.urls')),
    path('api/', include('forecasting.urls')),
    path('media/<path:path>', serve, {'document_root': settings.MEDIA_ROOT}),
]
