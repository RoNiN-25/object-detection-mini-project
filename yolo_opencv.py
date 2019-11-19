import cv2
import numpy as np
import zmq
import time


def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    


def number_points(a,boxes):
    
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
        cnt.append(np.array(L).reshape((-1,1,2)).astype(np.int32))
     
   for i in cnt:
        num = 0
        for n in range(a[0].shape[0]):
            if(cv2.pointPolygonTest(i, (a[0][n],a[1][n]), False) == -1.0):
                num=num+1
        number.append(num)
    
   if(number !=[]):
       k = number.index(max(number))
       return cnt[k].tolist()
   else:
       return None
       
        
   
    
scale = 0.00392
classes = None

context = zmq.Context()
socket = context.socket(zmq.REP)  #Reply socket. Gives info about something
socket.bind("tcp://*:5556")


prev1 = [0,0,0,0]
prev2 = [0,0,0,0]

lower_color = np.array([110, 50, 50])
upper_color = np.array([130 , 255 , 255])

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

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
    message = socket.recv()
    arr = np.frombuffer(message,np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    sent=0
    
    

    
    if(hhh%10==0):
        Width = image.shape[1]
        Height = image.shape[0]


        

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)
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
        fff = number_points(a,boxes)

        

        
        if(fff is not None):
            x1 = round(fff[0][0][0])
            y1 = round(fff[0][0][1])
            x2 = round(fff[2][0][0])
            y2 = round(fff[2][0][1])
            cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,255), 5)
            
            if(prev1[0]-x1>=20 and prev2[0]-prev1[0]>=25):
                print("Move right")
                #socket.send(b"R")
                state=11
                
            
            elif(prev1[0]-x1<-20 and prev2[0]-prev1[0]<-25):
                print("Move left")
                #socket.send(b"L")
                state=12
            
            elif((prev1[0]-prev1[2])*(prev1[1]-prev1[3])>1.2*(x1-x2)*(y1-y2) and (prev2[0]-prev2[2])*(prev2[1]-prev2[3])>0.8*(prev1[0]-prev1[2])*(prev1[1]-prev1[3])):
                print("Move forward")
                state=1
                #socket.send(b"F")
            elif((prev1[0]-prev1[2])*(prev1[1]-prev1[3])<0.8*(x1-x2)*(y1-y2) and (prev2[0]-prev2[2])*(prev2[1]-prev2[3])<1.2*(prev1[0]-prev1[2])*(prev1[1]-prev1[3])):
                print("Move backward")
                #socket.send(b"B")
                state=-1
            else:
                print("Stay")
                state=0
                #socket.send(b"St")
            
            prev2[0]=prev1[0]
            prev2[1]=prev1[1]
            prev2[2]=prev1[2]
            prev2[3]=prev1[3]
            
            prev1[0]=x1
            prev1[1]=y1
            prev1[2]=x2
            prev1[3]=y2
            
            if(state==11):
                socket.send(b"R")
                sent=1
            elif(state==12):
                socket.send(b"L")
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
            socket.send(b"OK")
            sent=1
        
        
        
        
        
        cv2.imshow("object detection", image)
    
        

    
   # end = time.time()
   # sum = sum+(end-start)

    
    hhh=hhh+1

    
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        socket.send(b"Stop")
        break
    #else:
        #socket.send(b"OK")


#cap.release()
cv2.destroyAllWindows()
