from rest_framework.viewsets import ModelViewSet 
from rest_framework.permissions import IsAuthenticated, AllowAny
from .models import Clients, BusinessRules
from users.models import UserInformation
from users.serializers import UserInformationSerializer
from .serializer import ClientsSerializer, BusinessRulesSerializer, ClientUserSerializer
from datetime import timezone
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.views import APIView
from django.core.exceptions import ObjectDoesNotExist
from django.utils.timezone import now
import string
from django.template.loader import render_to_string
from django.core.mail import  EmailMessage
from django.contrib.auth import get_user_model
from django.conf import settings
import threading


# Create your views here.
class ClientsViewSet(ModelViewSet):
    queryset = Clients.objects.all()
    serializer_class = ClientsSerializer
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        user = request.user
        try:
            user_info = UserInformation.objects.get(user=user)
            client = user_info.client

            serializer = self.get_serializer(client)
            return Response(serializer.data)
        except UserInformation.DoesNotExist:
            return Response({"detail": "User information not found"}, status=status.HTTP_404_NOT_FOUND)

    def perform_update(self, serializer):
        serializer.save(updated_at=timezone.now())

    @action(detail=False, methods=['patch'], url_path='patch-client')
    def update_client_info(self, request):
        user_profile = UserInformation.objects.filter(user=request.user).first()

        if not user_profile:
            return Response({"error": "No se encontró información de usuario."}, status=404)

        client = user_profile.client

        if not client:
            return Response({"error": "No se encontró un cliente asociado al perfil."}, status=404)

        serializer = self.get_serializer(client, data=request.data, partial=True)
        
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['put'], url_path='put-client')
    def update_client_info(self, request):
        user_profile = UserInformation.objects.filter(user=request.user).first()

        if not user_profile:
            return Response({"error": "No se encontró información de usuario."}, status=404)

        client = user_profile.client

        if not client:
            return Response({"error": "No se encontró un cliente asociado al perfil."}, status=404)

        client.address = request.data.get('address')
        client.category = request.data.get('category')
        client.country = request.data.get('country')
        client.state = request.data.get('state')
        client.phone = request.data.get('phone')

        client.save()
        return Response({'message': 'client_updated'}, status=status.HTTP_200_OK)
    

    def destroy(self, request, *args, **kwargs):
        client = self.get_object()
        client.delete()
        return Response({"detail": "Client deleted successfully"}, status=status.HTTP_204_NO_CONTENT)


class BusinessRulesModelViewSet(ModelViewSet):
    queryset = BusinessRules.objects.all()
    permission_classes = [IsAuthenticated]
    serializer_class = BusinessRulesSerializer

    def list(self, request, *args, **kwargs):
        client = request.user.userinformation.get().client
        queryset = self.get_queryset().filter(client=client).order_by('-created_at')
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
    
    def update(self, request, *args, **kwargs):
        fields = request.data.get('fields')
        update_rule = request.data.get('id')

        rule = BusinessRules.objects.get(id=update_rule)
        rule.fields = fields
        rule.updated_at = now()
        rule.save()

        return Response(data={'message': 'Regla actualizada exitosamente!'}, status=status.HTTP_200_OK)
    

## HELPERS ##
class ClientNameExistsAPIView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        client_name = request.GET.get('client')

        try:
            Clients.objects.get(name=client_name)
            return Response(data={'exists': True}, status=status.HTTP_200_OK)

        except ObjectDoesNotExist:
            return Response(data={'exists': False}, status=status.HTTP_200_OK)


## User clients ##
class GetClientsUsersAPIView(APIView):
    serializer_class = UserInformationSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return UserInformation.objects.filter(client=self.request.user.userinformation.get().client) 
    
    def get(self, request):
        data = self.get_queryset()
        serializer = UserInformationSerializer(data, many=True)
        return Response(data=serializer.data, status=status.HTTP_200_OK)


class ClientsUserView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = ClientUserSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            
            subject = "Te damos la bienvenida - DTA F&IO"
            
            context = {
                'username': serializer.validated_data['username'],
                'password': serializer.validated_data['password'],
                'user_admin': f'{request.user.first_name} {request.user.last_name}',
                'link': f"{settings.FRONTEND_URL}",
            }

            html_message = render_to_string('welcome_new_user.html', context)

            email = EmailMessage(
                subject,
                html_message,
                settings.DEFAULT_FROM_EMAIL,
                [serializer.validated_data['email']],
            )
            email.content_subtype = "html"

            # 🔹 Enviar el email en un hilo separado
            thread = threading.Thread(target=self.send_email, args=(email,))
            thread.start()

            return Response({"message": "El usuario fue creado exitosamente"}, status=201)

    @staticmethod
    def send_email(email):
        """Método estático para enviar el email en segundo plano"""
        email.send()


    def patch(self, request, user_id):
        try:
            user = get_user_model().objects.get(id=user_id)
        except get_user_model().DoesNotExist:
            return Response({"error": "El usuario no es miembro de tu empresa"}, status=404)

        serializer = ClientUserSerializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({"message": "El usaurio fue actualizado"}, status=200)
        return Response(serializer.errors, status=400)

    def delete(self, request, user_id):
        try:
            user = get_user_model().objects.get(id=user_id)
        except get_user_model().DoesNotExist:
            return Response({"error": "El usuario no es miembro de tu empresa"}, status=404)
        
        if request.user.id == user_id:
            return Response(
                {"error": "No puedes eliminar tu propia cuenta."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        user.delete()
        user.save()
        return Response(data={'message': 'Usuario eliminado exitosamente'}, status=status.HTTP_200_OK)