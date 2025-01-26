

import cv2
import uuid
import mediapipe as mp


face_detection = mp.solutions.face_detection.FaceDetection(0.6)


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


cap = cv2.VideoCapture(0)
count = 0

class_path = 'no_mask'
while True:
    _, frame = cap.read()
    img = frame.copy()
    try:
       
        x, y, w, h = get_detection(frame)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        
        crop_img = img[y:y+h, x:x+w]
        filename = "Data/"+class_path+"/"+str(uuid.uuid4())+".jpg"
        
        
        
        cv2.imwrite(filename, crop_img)
        cv2.imshow("frame", crop_img)
        count+=1
    except:
        pass
    if cv2.waitKey(1) == ord('q') or count>=500:
        break
    
cap.release()
cv2.destroyAllWindows()




