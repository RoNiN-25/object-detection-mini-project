#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:6565
#   Sends "Hello" to server, expects "World" back
#

import zmq
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(5,GPIO.OUT)
GPIO.setup(7,GPIO.OUT)
GPIO.setup(11,GPIO.OUT)
GPIO.setup(13,GPIO.OUT)

p1=GPIO.PWM(5,490)
p2=GPIO.PWM(7,490)
p3=GPIO.PWM(11,490)
p4=GPIO.PWM(13,490)

def forward():
#    GPIO.output(5,GPIO.LOW)
#    GPIO.output(7,GPIO.HIGH)
#    GPIO.output(11,GPIO.LOW)
#    GPIO.output(13,GPIO.HIGH)
    
    GPIO.output(5,GPIO.LOW)
    p2.start(75);
    GPIO.output(13,GPIO.HIGH)
    p4.start(75);
    
    GPIO.output(29,GPIO.LOW)
    GPIO.output(31,GPIO.HIGH)
    GPIO.output(33,GPIO.LOW)
    GPIO.output(35,GPIO.HIGH)
    
def backward():
#    GPIO.output(5,GPIO.HIGH)
#    GPIO.output(7,GPIO.LOW)
#    GPIO.output(11,GPIO.HIGH)
#    GPIO.output(13,GPIO.LOW)

    p1.start(75)
    GPIO.output(7,GPIO.LOW)
    p3.start(75)
    GPIO.output(13,GPIO.LOW)
    
    GPIO.output(29,GPIO.HIGH)
    GPIO.output(31,GPIO.LOW)
    GPIO.output(33,GPIO.HIGH)
    GPIO.output(35,GPIO.LOW)
    
def left():
#    GPIO.output(5,GPIO.LOW)
#    GPIO.output(7,GPIO.HIGH)
#    GPIO.output(11,GPIO.HIGH)
#    GPIO.output(13,GPIO.LOW)
    
    GPIO.output(5,GPIO.LOW)
    p2.start(65)
    p3.start(65)
    GPIO.output(13,GPIO.LOW)
    
    GPIO.output(29,GPIO.LOW)
    GPIO.output(31,GPIO.HIGH)
    GPIO.output(33,GPIO.HIGH)
    GPIO.output(35,GPIO.LOW)
    
def left1():
#    GPIO.output(5,GPIO.LOW)
#    GPIO.output(7,GPIO.HIGH)
#    GPIO.output(11,GPIO.HIGH)
#    GPIO.output(13,GPIO.LOW)
    
    GPIO.output(5,GPIO.LOW)
    p2.start(65)
    p3.start(65)
    GPIO.output(13,GPIO.LOW)
    
def right():
#    GPIO.output(5,GPIO.HIGH)
#    GPIO.output(7,GPIO.LOW)
#    GPIO.output(11,GPIO.LOW)
#    GPIO.output(13,GPIO.HIGH)
    
    p1.start(65)
    GPIO.output(7,GPIO.LOW)
    GPIO.output(11,GPIO.LOW)
    p4.start(65)
    
    GPIO.output(29,GPIO.HIGH)
    GPIO.output(31,GPIO.LOW)
    GPIO.output(33,GPIO.LOW)
    GPIO.output(35,GPIO.HIGH)
    
def right1():
#    GPIO.output(5,GPIO.HIGH)
#    GPIO.output(7,GPIO.LOW)
#    GPIO.output(11,GPIO.LOW)
#    GPIO.output(13,GPIO.HIGH)
    
    p1.start(65)
    GPIO.output(7,GPIO.LOW)
    GPIO.output(11,GPIO.LOW)
    p4.start(65)
    
def stop():
    GPIO.output(5,GPIO.LOW)
    GPIO.output(7,GPIO.LOW)
    GPIO.output(11,GPIO.LOW)
    GPIO.output(13,GPIO.LOW)
    p1.stop()
    p2.stop()
    p3.stop()
    p4.stop()
    
    GPIO.output(29,GPIO.LOW)
    GPIO.output(31,GPIO.LOW)
    GPIO.output(33,GPIO.LOW)
    GPIO.output(35,GPIO.LOW)
    


GPIO.setup(29,GPIO.OUT)
GPIO.setup(31,GPIO.OUT)
GPIO.setup(33,GPIO.OUT)
GPIO.setup(35,GPIO.OUT)



GPIO.output(5,GPIO.LOW)
GPIO.output(7,GPIO.LOW)
GPIO.output(11,GPIO.LOW)
GPIO.output(13,GPIO.LOW)

GPIO.output(29,GPIO.LOW)
GPIO.output(31,GPIO.LOW)
GPIO.output(33,GPIO.LOW)
GPIO.output(35,GPIO.LOW)

context = zmq.Context()
cap = cv2.VideoCapture(0)

#  Socket to talk to server
print("Connecting to server…")
socket = context.socket(zmq.REQ)  #Request socket. Asks for info about something
socket.connect("tcp://192.168.43.64:5558")


while(1):
    for i in range(0,10):
        _,frame = cap.read()
    strng = cv2.imencode('.jpg', frame)[1].tostring()
    socket.send(strng)

    #  Get the reply.
    message = socket.recv()
    if(message == b'Stop'):
        break
    elif(message == b'F'):
        print("Received reply ",message)
        forward()
        time.sleep(0.2)
        stop()
        #time.sleep(0.05)
        '''Move forward'''
    elif(message == b'B'):
        print("Received reply ",message)
        backward()
        time.sleep(0.2)
        stop()
        #time.sleep(0.05)
        '''Move Backward'''
    elif(message == b'L'):
        print("Received reply ",message)
        left()
        time.sleep(0.2)
        stop()
        #time.sleep(0.05)
        '''Move Left'''
    elif(message == b'R'):
        print("Received reply ",message)
        right()
        time.sleep(0.2)
        stop()
        #time.sleep(0.05)
        '''Move Right'''
    elif(message == b"R1"):
        print("Received reply ",message)
        right1()
        #time.sleep(0.2)
        #stop()
        
    elif(message == b"L1"):
        print("Received reply ",message)
        left1()
        #time.sleep(0.2)
        #stop()
        
    elif(message == b'St'):
        print("Received reply ",message)
        stop()
        #time.sleep(0.05)
        '''Stay''' 
    
    #print("Received reply ",message)
cap.release()
socket.close()
context.term()

