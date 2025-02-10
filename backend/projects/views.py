from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from rest_framework import status
from .serializers import ProjectSerializer
from .models import Projects
from users.models import UserInformation


class ProjectsViewSet(ModelViewSet):
    serializer_class = ProjectSerializer
    queryset = Projects.objects.all()
    permission_classes = [IsAuthenticated]

    def list(self, request, *args, **kwargs):
        client = request.user.userinformation.get().client
        queryset = self.get_queryset().filter(client=client).order_by('-created_at')
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, *args, **kwargs):
        project_serializer = self.get_serializer(data=request.data)

        if project_serializer.is_valid():
            project_serializer.save()
            return Response({'message': 'project_created', 'project': project_serializer.data},
                            status=status.HTTP_201_CREATED)

        else:
            return Response({'error': 'bad_request', 'logs': project_serializer.errors},
                            status=status.HTTP_400_BAD_REQUEST)

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        if serializer.is_valid():
            serializer.save()
            return Response({'message': 'project_updated', 'project': serializer.data}, status=status.HTTP_200_OK)
        return Response({'error': 'bad_request', 'logs': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def partial_update(self, request, *args, **kwargs):
        kwargs['partial'] = True
        return self.update(request, *args, **kwargs)
    
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        # user_profile = UserInformation.objects.filter(user=request.user).first()
        #if not instance.created_by.userinformation.client == user_profile.client:
        #    return Response({'error': 'No tiene permiso para eliminar este proyecto.'}, status=status.HTTP_403_FORBIDDEN)

        instance.delete()
        return Response({'message': 'project_deleted'}, status=status.HTTP_204_NO_CONTENT)
