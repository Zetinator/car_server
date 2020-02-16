from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

def index(request):
# http://{host}/
    """page to drop the car pictures...
    """
    try:
        context = {}
    except Exception as e:
        raise HttpResponse(status=500)
    return render(request, 'classifier/index.html', context)

@csrf_exempt
def predict(request):
    try:
        context = {}
    except Exception as e:
        raise HttpResponse(status=500)
    return HttpResponse("si...esta bien", status=200)