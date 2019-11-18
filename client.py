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
socket.connect("tcp://127.0.0.1:5555")


while(1):
    _,frame = cap.read()
    strng = cv2.imencode('.jpg', frame)[1].tostring()
    socket.send(strng)

    #  Get the reply.
    message = socket.recv()
    if(message == b'Stop'):
        break
    print("Received reply ",message)
cap.release()
socket.close()
context.term()