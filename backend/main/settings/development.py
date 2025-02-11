from .base import *

SECRET_KEY = os.environ.get('SECRET_KEY')

DEBUG = True

INSTALLED_APPS += ['dotenv']

ALLOWED_HOSTS = ['*']

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000"
]

CORS_ALLOW_ALL_ORIGINS = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': os.environ.get('DB_NAME'),
        'USER': os.environ.get('DB_USER'),
        'PASSWORD': os.environ.get('DB_PASSWORD'),
        'HOST': os.environ.get('DB_HOST'),
        'PORT': os.environ.get('DB_PORT'),
    }
}

EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')
FRONTEND_URL = os.environ.get('FRONTEND_URL')
LOAD_DATA_INFILE_DIR = 'C:\\ProgramData\\MySQL\\MySQL Server 8.0\\Uploads'