from rest_framework.serializers import ModelSerializer, ValidationError, CharField, IntegerField, EmailField
from django.contrib.auth import get_user_model
from clients.models import Clients, BusinessRules
from .models import UserInformation


class UserSerializer(ModelSerializer):
    client = CharField(required=True)

    class Meta:
        model = get_user_model()
        fields = ['username', 'password', 'email', 'first_name', 'last_name', 'client']

    def validate_username(self, value):
        User = get_user_model()
        if User.objects.filter(username=value).exists():
            raise ValidationError("username_already_exists")
        return value

    def validate_email(self, value):
        User = get_user_model()
        if User.objects.filter(email=value).exists():
            raise ValidationError("email_already_exists")
        return value

    def create(self, validated_data):
        client = validated_data.pop('client')
        User = get_user_model()

        user = User.objects.create_user(**validated_data)
        user.is_admin = True
        user.save()

        client = Clients.objects.create(name=client)
        UserInformation.objects.create(user=user, client=client)
        BusinessRules.objects.create(
            client=client,
            status=True, 
            fields=['family', 'region', 'client', 'salesman', 'sku'] 
        )
        return user


class UserInformationSerializer(ModelSerializer):
    user_id = IntegerField(source='user.id', read_only=True)
    username = CharField(source='user.username', read_only=True)
    first_name = CharField(source='user.first_name', required=False)
    last_name = CharField(source='user.last_name', required=False)
    email = EmailField(source='user.email', required=False)

    class Meta:
        model = UserInformation
        fields = [
            'user_id', 'first_name', 'last_name', 'username', 'email', 'phone',
            'address', 'birth_date', 'client'
        ]

    def update(self, instance, validated_data):
        user_data = validated_data.pop('user', {})
        user = instance.user

        user.username = user_data.get('username', user.username)
        user.email = user_data.get('email', user.email)
        user.first_name = user_data.get('first_name', user.first_name)
        user.save()

        instance.phone = validated_data.get('phone', instance.phone)
        instance.address = validated_data.get('address', instance.address)
        instance.birth_date = validated_data.get('birth_date', instance.birth_date)
        instance.client = validated_data.get('client', instance.client)
        instance.save()

        return instance

    def to_representation(self, instance): 
        return {
            "user_id": instance.user.id,
            "first_name": instance.user.first_name,
            "last_name": instance.user.last_name,
            "username": instance.user.username,
            "email": instance.user.email,
            "phone": instance.phone,
            "address": instance.address,
            "birth_date": instance.birth_date,
            "client": instance.client.name if instance.client else None,
            "position": instance.position
        }
