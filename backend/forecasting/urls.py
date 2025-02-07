from .views import ForecastScenarioViewSet, AnalyticsViewsForScenarios, ColaborationViews, ListModelsInformationAPIView, SetBestModelAPIView, KpisViews, GetScenarioNameAPIView
from rest_framework.routers import DefaultRouter
from django.urls import path

router = DefaultRouter()
router.register('forecast', ForecastScenarioViewSet, basename='forecast')

get_analytics = AnalyticsViewsForScenarios.ForecastAnalyticsAPIView.as_view()
get_cluster_table = AnalyticsViewsForScenarios.ClusterDataTable.as_view()
get_reports = AnalyticsViewsForScenarios.ErrorReportsView.as_view()
colaborate = ColaborationViews.ColaborationForecastAPIView.as_view()
get_colaboration_data = ColaborationViews.ListColaborationDataAPIView.as_view()
get_dates = ColaborationViews.GetPredictedDatesAPIView.as_view()
get_chart_colaboraiton = ColaborationViews.GetColaborationChartAPIView.as_view()
get_models_data = ListModelsInformationAPIView.as_view()
select_best_model = SetBestModelAPIView.as_view()
year_kpi = KpisViews.KpisByYearAPIView.as_view()
kpi = KpisViews.KpisByGroupAPIView.as_view()
get_name = GetScenarioNameAPIView.as_view()

urlpatterns = [
    ## COLABORATIONS URLS ##
    path('forecast/colaboration/sales/update/',colaborate, name='colaborate'),
    path('forecast/colaboration/sales/update/',colaborate, name='colaborate'),
    path('forecast/colaboration/sales/',get_colaboration_data, name='get_colaboration_data'),
    path('forecast/colaboration/chart/',get_chart_colaboraiton, name='get_chart_colaboraiton'),

    ## KPIS URLS ##
    path('forecast/kpis/year/', year_kpi, name='year_kpi'),
    path('forecast/kpis/', kpi, name='kpi'),

    ## ANALYSIS URLS ##
    path('forecast/analysis/dates/', get_dates, name='get_dates'),
    path('forecast/analysis/models/', get_models_data, name='get_models_data'),
    path('forecast/analysis/models/select/', select_best_model, name='select_best_model'),
    path('forecast/analysis/sales/', get_analytics, name='get_analytics'),
    path('forecast/analysis/cluster/', get_cluster_table, name='get_cluster_table'),
    path('forecast/analysis/reports/', get_reports, name='get_reports'),

    ## HELPERS ##
    path('forecast/helpers/name/', get_name, name='get_name'),

] + router.urls
