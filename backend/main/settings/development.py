from .base import *
from dotenv import load_dotenv

env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

SECRET_KEY = os.getenv('SECRET_KEY')

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
        'NAME': os.getenv('DB_NAME'),
        'USER': os.getenv('DB_USER'),
        'PASSWORD': os.getenv('DB_PASSWORD'),
        'HOST': os.getenv('DB_HOST'),
        'PORT': os.getenv('DB_PORT'),
    }
}

EMAIL_HOST_USER = os.getenv('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.getenv('EMAIL_HOST_PASSWORD')
FRONTEND_URL = os.getenv('FRONTEND_URL')
