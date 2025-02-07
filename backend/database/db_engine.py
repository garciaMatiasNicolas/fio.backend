from sqlalchemy import create_engine
from django.conf import settings

# Configuración de la base de datos
DATABASES = settings.DATABASES['default']
DB_NAME = DATABASES['NAME']
DB_USER = DATABASES['USER']
DB_PASSWORD = DATABASES['PASSWORD']
DB_HOST = DATABASES['HOST']
DB_PORT = DATABASES['PORT']

# Crear la URL de conexión de SQLAlchemy
DB_URL = f'mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
# Crear el engine de SQLAlchemy
engine = create_engine(DB_URL, echo=False)
