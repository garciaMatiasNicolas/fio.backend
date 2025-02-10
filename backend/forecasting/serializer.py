from rest_framework import serializers
from .models import Scenario, PredictedSale
from users.models import UserInformation
from projects.models import Projects
from file.models import File


class ForecastScenarioSerializer(serializers.ModelSerializer):
    project = serializers.CharField(write_only=True)

    class Meta:
        model = Scenario
        exclude = ('runned_by', 'created_at', 'client', 'historical_file')
    
    def validate(self, data):
        request = self.context.get('request')
        user = request.user
        
        try:
            user_info = user.userinformation.get()
            client = user_info.client
            if not client:
                raise serializers.ValidationError("missing_client")
        except UserInformation.DoesNotExist:
            raise serializers.ValidationError("missing_userinformation")

        project_name = data.get('project')
        project = Projects.objects.filter(name=project_name, client=client).first()
        if not project:
            raise serializers.ValidationError("project_not_exists")

        if Scenario.objects.filter(name=data['name'], client=client, project__name=project_name).exists():
            raise serializers.ValidationError("scenario_name_already_in_use")
        
        data['project'] = project

        return data
    
    def create(self, validated_data):
        request = self.context.get('request')
        user = request.user
        user_info = user.userinformation.get()
        client = user_info.client

        project = validated_data.pop('project')
        file = File.objects.filter(project=project, file_type='historical').first()
        if not file:
            raise serializers.ValidationError("historical_file_missing")

        scenario = Scenario.objects.create(
            name=validated_data['name'],
            runned_by=user,
            client=client,
            project=project,
            historical_file=file,
            pred_p=validated_data.get('pred_p'),
            test_p=validated_data.get('test_p'),
            error_p=validated_data.get('error_p'), 
            replace_negatives=validated_data.get('replace_negatives'),
            seasonal_periods=validated_data.get('seasonal_periods'),
            error_type=validated_data.get('error_type'),
            additional_params=validated_data.get('additional_params'),
            models=validated_data.get('models'),
            explosive=validated_data.get('explosive'),
            filter_products=validated_data.get('filter_products'),
            detect_outliers=validated_data.get('detect_outliers'),
            is_daily=True if project.periods_per_year == 365 else False
        )

        return scenario

    def to_representation(self, instance):
        return {
            "id": instance.id,
            "name": instance.name,
            "runned_by": f'{instance.runned_by.first_name} {instance.runned_by.last_name}',
            "created_at": instance.created_at,
            "pred_p": instance.pred_p,
            "test_p": instance.test_p,
            "error_p": instance.error_p,
            "models": instance.models,
            "filter_products": instance.filter_products
        }


class PredictedSaleSerializer(serializers.ModelSerializer):

    class Meta:
        model = PredictedSale
        fields = '__all__'