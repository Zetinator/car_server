from django.contrib import admin
from django.urls import include, path

app_name = 'classifier'
urlpatterns = [
    path('admin/', admin.site.urls),
    path('classifier/', include('classifier.urls')),
]
