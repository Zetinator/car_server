from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

def index(request):
# http://{host}/classifier
    """page to drop the car pictures...
    """
    try:
        context = {}
    except Exception as e:
        raise HttpResponse(status=500)
    return render(request, 'classifier/index.html', context)

@csrf_exempt
def predict(request):
# http://{host}/classifier
    """page to load the prediction...
    """
    try:
        context = {}
        context['name'] = 'erick quiere mucho a su Marion :('
        if request.method == 'POST':
            if 'image' not in request.FILES['file'].content_type:
                return HttpResponse(status=415)
            # prediction using our network starts here...
            context['name'] = request.FILES['file'].name
            context['prediction'] = 'Okay you got me... I dont know that car :('
            img = request.FILES['file'].read()
            print(context)
    except Exception as e:
        raise HttpResponse(status=500)
    return render(request, 'classifier/prediction.html', context)
