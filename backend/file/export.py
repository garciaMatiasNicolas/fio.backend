from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.http import HttpResponse
from rest_framework.exceptions import NotFound
from django.http import FileResponse
import pandas as pd
from .models import File
import os


class ExportToCsvAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        columns = request.data.get('columns') 
        data = request.data.get('data')     
        file_name = request.data.get('file_name')

        df = pd.DataFrame(data, columns=columns).astype('str')
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={file_name}.csv'

        df.to_csv(response, index=False, sep=";", encoding='utf-8')
        return response


class DownloadFileAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, file_id, *args, **kwargs):
        try:
            file_instance = File.objects.get(id=file_id)

            file_path = file_instance.file_path

            if not os.path.exists(file_path):
                raise NotFound("File not found")

            response = FileResponse(open(file_path, 'rb'))
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response

        except File.DoesNotExist:
            raise NotFound("File not found")
