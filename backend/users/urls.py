from django.urls import path
from .views import UserInformationModelViewSet, SignUpView, TokenValidationView, PasswordResetView, PasswordResetConfirmView, EmailFromLandingAPIView, UsernameOrEmailExistsAPIView
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView
)

signup = SignUpView.as_view()
router = DefaultRouter()
token_validation = TokenValidationView.as_view()

router.register('userinformation', UserInformationModelViewSet, basename='user_info_router')

urlpatterns = [ 
    path('users/signup/', signup, name='create_user'),
    path('users/token-validation/', token_validation, name='send_token'),
    path('users/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('users/token-refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('users/password-reset/', PasswordResetView.as_view(), name='password_reset'),
    path('users/password-reset-confirm/<uidb64>/<token>/', PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('users/helpers/validate/',UsernameOrEmailExistsAPIView.as_view(), name='username_or_email_helper' ),
    path('web/form/', EmailFromLandingAPIView.as_view(), name='set_demo_form'),
] + router.urls