from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from .serializers import UserSerializer, UserInformationSerializer
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import UserInformation
from rest_framework.decorators import action
from django.contrib.auth import get_user_model
from django.core.mail import send_mail, EmailMessage
import random
import string
from django.utils.cache import caches
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, smart_str
from django.conf import settings
from django.template.loader import render_to_string
from django.core.exceptions import ObjectDoesNotExist


cache = caches['default']
User = get_user_model()

class SignUpView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            return Response({"message": "user_created"}, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class TokenValidationView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        email = request.GET.get('email')
        user = User.objects.get(email=email)
        
        token = ''.join(random.choices(string.digits, k=6))
        cache.set(f"validation_token_{user.id}", token, timeout=90)

        context = {
            'token': token,
        }

        subject = "Token de validación - DTA F&IO"
        html_message = render_to_string('token_validation.html', context)

        mail = EmailMessage(
            subject,
            html_message,
            "noreply@example.com",
            [email],
        )
        mail.content_subtype = "html"
        mail.send()

        return Response({"message": "token_sent"}, status=status.HTTP_200_OK)

    def post(self, request):
        token = request.data.get('token')
        email = request.data.get('email')
        user = User.objects.get(email=email)

        if isinstance(token, list):
            token = ''.join(token)  
        
        cached_token = cache.get(f"validation_token_{user.id}")

        if token != cached_token:
            user.is_active = False
            user.save()
            return Response({"detail_error": "El token no es válido. Intente nuevamente"}, status=status.HTTP_400_BAD_REQUEST)

        user.is_active = True 
        user.save()
        cache.delete(f"validation_token_{user.id}")

        return Response({"detail": "Token validado correctamente."}, status=status.HTTP_200_OK)


class UserInformationModelViewSet(ModelViewSet):
    permission_classes = [IsAuthenticated]
    serializer_class = UserInformationSerializer
    queryset = UserInformation.objects.all()
    
    def list(self, request, *args, **kwargs):
        user_profile = UserInformation.objects.filter(user=request.user).first()
        
        if not user_profile:
            return Response({"error": "No se encontró información de usuario."}, status=404)
        
        serializer = self.get_serializer(user_profile)
        return Response(serializer.data)
    
    @action(detail=False, methods=['patch'], url_path='patch-profile')
    def update_my_profile(self, request):
        user_profile = UserInformation.objects.filter(user=request.user).first()
        
        if not user_profile:
            return Response({"error": "No se encontró información de usuario."}, status=404)

        serializer = self.get_serializer(user_profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['put'], url_path='put-profile')
    def _update_profile_conf(self, request):
        user_profile = UserInformation.objects.filter(user=request.user).first()

        if not user_profile:
            return Response({"error": "No se encontró información de usuario."}, status=404)
        
        user_profile.address = request.data.get('address')
        user_profile.phone = request.data.get('phone')
        user_profile.birth_date = request.data.get('birth_date')
        user_profile.position = request.data.get('position')

        user_profile.save()

        return Response({'message': 'succeed'}, status=status.HTTP_200_OK)


class PasswordResetView(APIView):
    permission_classes=[AllowAny]

    """
    Step 1: Request a password reset link.
    """
    def get(self, request):
        email = request.GET.get("email")
        if not email:
            return Response({"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response({"error": "No se encontro un usuario con el mail ingresado"}, status=status.HTTP_404_NOT_FOUND)
        
        # Generate password reset token
        token_generator = PasswordResetTokenGenerator()
        token = token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        
        # Create reset link
        reset_link = f"{settings.FRONTEND_URL}/reset-password/{uid}/{token}/"
        
        context = {
            'token': token,
            'reset_link': reset_link,
        }
        
        subject = "Restablecimiento de contraseña - DTA F&IO"
        html_message = render_to_string('reset_password.html', context)
        
        # Send email
        email = EmailMessage(
            subject,
            html_message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
        )
        email.content_subtype = "html"  # This is to send the email as HTML.
        email.send()

        return Response({"message": "Password reset link sent to your email."}, status=status.HTTP_200_OK)


class PasswordResetConfirmView(APIView):
    permission_classes=[AllowAny]
    
    """
    Step 2: Confirm and reset the password using the token.
    """
    def post(self, request, uidb64, token):
        new_password = request.data.get("new_password")
        
        try:
            uid = smart_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (User.DoesNotExist, ValueError, TypeError):
            return Response({"error": "Token Ínvalido"}, status=status.HTTP_400_BAD_REQUEST)
        
        token_generator = PasswordResetTokenGenerator()
        if not token_generator.check_token(user, token):
            return Response({"error": "El token ha expirado. Intente nuevamente"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Set new password
        user.set_password(new_password)
        user.save()
        
        return Response({"message": "ok"}, status=status.HTTP_200_OK)


class EmailFromLandingAPIView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        data = request.data

        email = data.get("email")
        message = data.get("message")
        full_name = data.get("full_name")
        phone = data.get("phone", None)
        client = data.get("client")

        subject = "DEMO CLIENTES - WEB DTA F&IO - NUEVA CONSULTA"
        email_message = (
            f'''
                Has recibido una consulta desde la web de DTA-F&IO.
                La persona {full_name} de la empresa {client} ha solicitado una demo. Sus datos de contacto son:
                    - Nombre completo: {full_name}
                    - Nombre de su empresa: {client}
                    - Email: {email}
                    - Mensaje adicional: {message if message is not None else "No especificado"}
                    - Teléfono: {phone if phone is not None else "No especificado"}
            '''
        )

        try:
            send_mail(
                subject=subject,
                message=email_message,
                from_email='',  
                recipient_list=['garciamatias159@gmail.com'],  
            )
            return Response({"message": "Su consulta fue enviada correctamente."}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": f"No se pudo enviar el email. Detalles: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


## HELPERS ##
class UsernameOrEmailExistsAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        username = request.GET.get('username')
        email = request.GET.get('email')

        try:
            if email:
                User.objects.get(email=email)
                return Response(data={'exists': True}, status=status.HTTP_200_OK)

            if username:
                User.objects.get(username=username)
                return Response(data={'exists': True}, status=status.HTTP_200_OK)

        except ObjectDoesNotExist:
            return Response(data={'exists': False}, status=status.HTTP_200_OK)