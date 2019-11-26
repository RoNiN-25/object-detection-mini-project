import cv2
import numpy as np
import zmq
import time


def get_output_layers(net):   #FUnction for getting the output layers

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):  # draws a bounding box around all preictions

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    


def number_points(a,boxes):  #Function for finidng the box with max number of color points in it
    
   cnt = []
   number = []
    
    
   for i in boxes:
        x1 = i[0]
        y1 = i[1]
        x2 = x1+i[2]
        y2 = y1
        x3 = x2
        y3 = y2+i[3]
        x4 = x1
        y4 = y3
        L = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        cnt.append(np.array(L).reshape((-1,1,2)).astype(np.int32))  #makes a contour and appends it to a list
     
   for i in cnt:
        num = 0
        for n in range(a[0].shape[0]):
            if(cv2.pointPolygonTest(i, (a[0][n],a[1][n]), False) == -1.0):
                num=num+1
        number.append(num)
    
   print(number) 
   if(number !=[]):
       k = number.index(min(number))
       return (cnt[k].tolist(),cnt[k])  #returns the contour
   else:
       return (None,None)
       
        
   
    
scale = 0.00392
classes = None

context = zmq.Context()
socket = context.socket(zmq.REP)  #Reply socket. Gives info about something
socket.bind("tcp://*:5558")


lower_color = np.array([110, 50, 50])   #For blue
upper_color = np.array([130 , 255 , 255])

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)  #Sets processing to GPU

hhh=0
sum=0.0
state = 0 
sent = 1
#cap = cv2.VideoCapture(1)
while (1):
   # start = time.time()
    #_,image = cap.read()
    if(sent!=1):
        socket.send(b" ")
    message = socket.recv()   # Receive image
    arr = np.frombuffer(message,np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)  #Decode the image and add color to it
    sent=0
    
    

    
    if(hhh%1==0):
        Width = image.shape[1]
        Height = image.shape[0]
        
        print(Width)
        print(Height)
        blurred = cv2.GaussianBlur(image, (11, 11), 0)
                

        blob = cv2.dnn.blobFromImage(blurred, scale, (192,192), (0,0,0), True, crop=False)   #Generate a blob

        net.setInput(blob) 

        outs = net.forward(get_output_layers(net))  #Gets all predictions
        
        #blurred = cv2.GaussianBlur(image, (11, 11), 0)
        
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)    #Generate mask
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        res = cv2.bitwise_and(image,image, mask= mask)  

        

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if(class_id == 0):
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
       
        
        
        
        #cv2.imshow('frame',image)
        cv2.imshow('mask',mask)
        #cv2.imshow('res',res)
        

        a=np.where(mask==255)
        fff,cnt = number_points(a,boxes)

        

        #cv2.rectangle(image, (150,40), (490,440), (0,255,255), 7) #Reference box
        cv2.circle(image,(350,280),150,(0,0,0),2)  #Reference circle
        
        if(fff is not None and cnt is not None):
            x1 = round(fff[0][0][0])  #Corner points
            y1 = round(fff[0][0][1])
            x2 = round(fff[2][0][0])
            y2 = round(fff[2][0][1])
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,255), 5)
            
            
            
            ((x_center,y_center),r)=cv2.minEnclosingCircle(cnt) #Generate the enclosing circle
            print(((x_center,y_center)))
            cv2.circle(image,(int(x_center),int(y_center)),int(r),(0,255,255),2)
            
            

            
            if(185-r>15):
                print("Move forward")
                state=1
                
            elif(185-r<-15):
                print("Move backward")
                state=-1
                
                
            elif(x_center-320<-50):
                print("Move Left")
                state=11
                
                
            elif(x_center-320>50):
                print("Move right")
                state=12
                
            else:
                print("Stay")
                state=0
            
            
            if(state==11):
                socket.send(b"R")
                sent=1
            elif(state==111):
                socket.send(b"R1")
                sent=1
            elif(state==12):
                socket.send(b"L")
                sent=1
            elif(state==121):
                socket.send(b"L1")
                sent=1
            elif(state==1):
                socket.send(b"F")
                sent=1
            elif(state==-1):
                socket.send(b"B")
                sent=1
            elif(state==0):
                socket.send(b"St")
                sent=1
                    
        else:
            print("No object to follow")
            socket.send(b"St")
            sent=1
        
        
        
        
        
        cv2.imshow("object detection", image)
    
        

    
   # end = time.time()
   # sum = sum+(end-start)

    
    hhh=hhh+1

    
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break


#cap.release()
if(sent==1):
    j = socket.recv()
socket.send(b"Stop")
cv2.destroyAllWindows()
