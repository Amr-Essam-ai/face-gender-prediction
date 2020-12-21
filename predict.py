import base64
import json
import falcon
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import cgi  
import falcon
import os,cv2

def convert_image(images,w=299,h=299):
    images_j=[]
    c=3
    for i in range (len(images)):
        img = cv2.imdecode(np.fromstring(images["image"+str(i)].file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img =cv2.resize(img, (w,h))
        img = np.expand_dims(img, axis =0)
        if len(img.shape) <3:c=1
        data = (img / 255).reshape(1, w, h, c)
        images_j.append(data)
    return np.vstack(images_j)


class PredictResource(object):

    def __init__(self, model):
        self.model = model

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = 'GENDER'

    def on_post(self, req, resp):
        data={}
        form = cgi.FieldStorage(fp=req.stream, environ=req.env)
       
        images=convert_image(form,w=198,h=198)

        result=self.model.predict_label(images)

        data["result"]=result


        # predicted_data = self.model.predict_classes(data)[0]
        # output = {'prediction': str(predicted_data)}
        # resp.status = falcon.HTTP_200
        resp.body = json.dumps(data, ensure_ascii=False)