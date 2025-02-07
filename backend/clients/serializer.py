from rest_framework import serializers
from .models import Clients, BusinessRules
from users.models import UserInformation
from datetime import datetime
from django.contrib.auth import get_user_model


class ClientsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Clients
        exclude = ('updated_at', 'created_at',)

    def validate(self, attrs):
        name = attrs.get('name')
        if Clients.objects.filter(name=name).exists():
            raise serializers.ValidationError("Client name already in use")
        return attrs

    def create(self, validated_data):
        validated_data['admin'] = self.context['request'].user
        client = Clients.objects.create(**validated_data)

        return client

    def update(self, instance, validated_data):
        validated_data['updated_at'] = datetime.now()
        return super().update(instance, validated_data)

    def to_representation(self, instance):
        return {
            "id": instance.id,
            "name": instance.name,
            "address": instance.address,
            "phone": instance.phone,
            "category": instance.category,
            "country": instance.country,
            "state": instance.state,
        }


class BusinessRulesSerializer(serializers.ModelSerializer):
    class Meta:
        model = BusinessRules
        exclude = ('updated_at', 'created_at',)

    def to_representation(self, instance):
        return {
            "id": instance.id,
            "created_at": instance.created_at,
            "updated_at": instance.updated_at,
            "fields": instance.fields,
        }
    

class ClientUserSerializer(serializers.ModelSerializer):

    class Meta:
        model = get_user_model()
        fields = ['username', 'password', 'email', 'first_name', 'last_name']

    def create(self, validated_data):
        user = get_user_model().objects.create_user(**validated_data)
        user.is_active = True
        user.is_admin = False
        user.save()
        client = self.context['request'].user.userinformation.get().client 

        UserInformation.objects.create(user=user, client=client)

        return user

    def update(self, instance, validated_data):
        # Actualizar los campos del usuario
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.save()

        return instance