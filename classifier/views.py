import io

from django.template import loader
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt

from .network import Classifier
from .classes import classes

# initialize the network for predictions
model = Classifier()
model.load_checkpoint('./classifier/efficientnet_b0_e100')

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
# http://{host}/classifier/predict
    """page to load the prediction...
    """
    try:
        context = {}
        context['prediction'] = 'Okay you got me... I dont know that car :('
        # no file? nothing to do here...
        if request.method == 'POST' and 'file' in request.FILES:
            # somehow not an image... return
            if 'image' not in request.FILES['file'].content_type:
                return HttpResponse(status=415)
            # prediction using our network starts here...
            context['name'] = request.FILES['file'].name
            img = request.FILES['file'].read()
            stream = io.BytesIO(img)
            prediction, second = model.predict(stream, show_probability=True)
            context['prediction'] = prediction
            context['second'] = second
    except Exception as e:
        raise HttpResponse(status=500)
    return render(request, 'classifier/prediction.html', context)
