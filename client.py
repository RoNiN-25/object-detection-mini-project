#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#

import zmq
import cv2
import numpy as np

context = zmq.Context()
cap = cv2.VideoCapture(0)

#  Socket to talk to server
print("Connecting to server…")
socket = context.socket(zmq.REQ)  #Request socket. Asks for info about something
socket.connect("tcp://localhost:5558")


while(1):
    _,frame = cap.read()
    strng = cv2.imencode('.jpg', frame)[1].tostring()
    socket.send(strng)

    #  Get the reply.
    message = socket.recv()
    if(message == b'Stop'):
        break
    elif(message == b'F'):
        print("Received reply ",message)
        '''Move forward'''
    elif(message == b'B'):
        print("Received reply ",message)
        '''Move Backward'''
    elif(message == b'L'):
        print("Received reply ",message)
        '''Move Left'''
    elif(message == b'R'):
        print("Received reply ",message)
        '''Move Right'''
    elif(message == b'St'):
        print("Received reply ",message)
        '''Stay''' 
    
    #print("Received reply ",message)
cap.release()
socket.close()
context.term()
