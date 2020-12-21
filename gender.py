import tensorflow as tf

import numpy as np 
import cv2
import tensorflow.keras
from  tensorflow.keras.models import load_model
import pandas as pd 
    
from sklearn.metrics import  f1_score


def f1_micro(y_true, y_pred):

    return f1_score(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1), average = 'micro', zero_division=0)
    
def tf_f1_micro(y_true, y_pred):

    return tf.py_function(f1_micro, inp=[y_true, y_pred], Tout=tf.float32)




class model_g:
    def __init__(self):
        self.model_g=self.load_model()

    def load_model(self):
        print("load model")
        model1=tf.keras.models.load_model('model/gender_model_v3.h5',custom_objects={'tf': tf,'tf_f1_micro':tf_f1_micro})
        x = tf.keras.layers.Activation('softmax',name='gender_out')(model1.layers[-2].output)
        gender_model = tf.keras.models.Model(inputs=model1.input,outputs =x)
        return gender_model


    def pred_transform(self,pred):

        """ pred_transform convert pred  to label and confidence

        Parameters
        ----------
        pred : tensor
        (None,2)
    

        Returns
        -------
        results
        a dict of gender label and confidence
        {gender:{'label':[],'confidence':[]}}
        """

  
        gender_label,gender_confidence = np.argmax(pred, axis=1),np.int16(np.max(pred*100, axis=1))
        
        gender_label=np.where(gender_label==0, 'male', gender_label) 
        
        gender_label=np.where(gender_label=='1', 'female', gender_label) 

        results={'gender':{'label':gender_label.tolist(),

        'confidence':gender_confidence.tolist()}}
        return results 

    def predict_label(self,images):
        """ predict model

        Parameters
        ----------
        images : np.array
        
    

        Returns
        -------
        results
        a dict of gender label and confidence
        {gender:{'label':[],'confidence':[]}}
        """

        pred=self.model_g.predict(images)
        pred_df=self.pred_transform(pred)
        return pred_df
    
    
   
 




