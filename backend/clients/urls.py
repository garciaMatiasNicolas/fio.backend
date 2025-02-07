from .views import ClientsViewSet, ClientNameExistsAPIView, BusinessRulesModelViewSet, GetClientsUsersAPIView, ClientsUserView
from rest_framework.routers import DefaultRouter
from django.urls import path

router_clients = DefaultRouter()

router_clients.register('clients', ClientsViewSet, basename='clients_routes')
router_clients.register('businessrules', BusinessRulesModelViewSet, basename='business_rules')

urlpatterns = [ 
    path('clients/helpers/validate/', ClientNameExistsAPIView.as_view(), name='helper_client_name'),
    path('clients/users/create/', ClientsUserView.as_view(), name='create_client_user'),
    path('clients/users/update/<int:user_id>/', ClientsUserView.as_view(), name='update_client_user'),
    path('clients/users/delete/<int:user_id>/', ClientsUserView.as_view(), name='update_client_user'),
    path('clients/users/', GetClientsUsersAPIView.as_view(), name='list_clients_users'),    
] + router_clients.urls