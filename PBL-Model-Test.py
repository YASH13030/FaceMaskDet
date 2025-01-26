

import cv2 
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from keras.models import load_model


model = load_model('model.h5')



face_detection = mp.solutions.face_detection.FaceDetection()





def get_detection(frame):

    height, width, channel = frame.shape

    

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    result = face_detection.process(imgRGB)


    try:
        for count, detection in enumerate(result.detections):

           

        
            
            
            box = detection.location_data.relative_bounding_box

            x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width), int(box.height*height)
            
    
    except:
        pass

    return x, y, w, h

CATEGORIES = ['no_mask', 'mask']
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    img = frame.copy()
    try:
        x, y, w, h = get_detection(frame)
        
        crop_img = img[y:y+h, x:x+w]
        
        crop_img = cv2.resize(crop_img, (100, 100))
        
        crop_img = np.expand_dims(crop_img, axis=0)
        
       
        prediction = model.predict(crop_img)
        print(prediction)
        index = np.argmax(prediction)
        res = CATEGORIES[index]
        if index == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.8, color, 2, cv2.LINE_AA)

    except:
        pass
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()



