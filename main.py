import tensorflow as tf

import os
import falcon
from predict import PredictResource
from falcon_cors import CORS    
import numpy as np 
import cv2
from gender import model_g
from PIL import Image

api = application = falcon.API()

def load_trained_model():
    global model
    model=model_g()
    return model

predict = PredictResource(model=load_trained_model())
api.add_route('/gender/', predict)

