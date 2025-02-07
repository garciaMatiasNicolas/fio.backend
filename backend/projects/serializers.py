from rest_framework.serializers import ModelSerializer, ValidationError
from .models import Projects
from users.models import UserInformation


class ProjectSerializer(ModelSerializer):
    class Meta:
        model = Projects
        exclude = ('created_at', 'status', 'created_by', 'client')
    
    def validate(self, data):
        request = self.context.get('request')
        user = request.user

        try:
            user_info = user.userinformation.get() 
            if not user_info.client:
                raise ValidationError("missing_client")
        except UserInformation.DoesNotExist:
            raise ValidationError("missing_userinformation")

        client = user_info.client

        if Projects.objects.filter(name=data['name'], client=client).exists():
            raise ValidationError("project_name_already_in_use")

        return data

    def create(self, validated_data):
        request = self.context.get('request')
        user = request.user

        try:
            client = user.userinformation.get().client
        except UserInformation.DoesNotExist:
            raise ValidationError("missing_userinformation")
        except AttributeError:
            raise ValidationError("missing_client")

        project = Projects.objects.create(
            name=validated_data['name'],
            status=validated_data.get('is_active', True),
            created_by=user,
            client=client
        )

        return project

    def to_representation(self, instance):
        return {
            'id': instance.id,
            'project_name': instance.name,
            'created_at': instance.created_at,
            'status': instance.status,
            'client': instance.client.name if instance.client else None,
            'user_owner': f'{instance.created_by.first_name} {instance.created_by.last_name}' if instance.created_by else None
        }