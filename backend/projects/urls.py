from rest_framework.routers import DefaultRouter
from .views import ProjectsViewSet

router_projects = DefaultRouter()

router_projects.register('projects', ProjectsViewSet, basename='projects_routes')

urlpatterns = router_projects.urls