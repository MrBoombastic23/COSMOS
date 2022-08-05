import cv2 as cv2
from cv2 import minAreaRect
import matplotlib.pyplot as plt
import numpy as np
from LaneTrackingFunctions import *
import pyvesc as pyvesc
import depthai as dai

VESC_SERIAL = "/dev/tty.usbmodem3041"
VESC_BAUD = 115200


class VESC:
''' 
VESC Motor controler using pyvesc
This is used for most electric skateboards.
inputs: serial_port---- port used communicate with vesc. for linux should be something like /dev/ttyACM1
has_sensor=False------- default value from pyvesc
start_heartbeat=True----default value from pyvesc (I believe this sets up a heartbeat and kills speed if lost)
baudrate=115200--------- baudrate used for communication with VESC
timeout=0.05-------------time it will try before giving up on establishing connection
percent=.2--------------max percentage of the dutycycle that the motor will be set to
outputs: none
uses the pyvesc library to open communication with the VESC and sets the servo to the angle (0-1) and the duty_cycle(speed of the car) to the throttle (mapped so that percentage will be max/min speed)
Note that this depends on pyvesc, but using pip install pyvesc will create a pyvesc file that
can only set the speed, but not set the servo angle. 
Instead please use:
pip install git+https://github.com/LiamBindle/PyVESC.git@master
to install the pyvesc library
'''
def __init__(self, serial_port, percent=.2, has_sensor=False, start_heartbeat=True, baudrate=115200, timeout=0.05, steering_scale = 1.0, steering_offset = 0.0 ):
try:
import pyvesc
except Exception as err:
print("\n\n\n\n", err, "\n")
print("please use the following command to import pyvesc so that you can also set")
print("the servo position:")
print("pip install git+https://github.com/LiamBindle/PyVESC.git@master")
print("\n\n\n")
#time.sleep(1)
raise
assert percent <= 1 and percent >= -1,'\n\nOnly percentages are allowed for MAX_VESC_SPEED (we recommend a value of about .2) (negative values flip direction of motor)'
self.steering_scale = steering_scale
self.steering_offset = steering_offset
self.percent = percent
try:
self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)
except Exception as err:
print("\n\n\n\n", err)
print("\n\nto fix permission denied errors, try running the following command:")
print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")
time.sleep(1)
raise
def run(self, angle, throttle):
self.v.set_servo((angle * self.steering_scale) + self.steering_offset)
self.v.set_duty_cycle(throttle*self.percent)

vesc = VESC(serial_port=VESC_SERIAL)

pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(1500, 800) # resolution here
camRgb.setInterleaved(False)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

xoutRgb = pipeline.create(dai.node.XLinkOut)
camRgb.preview.link(xoutRgb.input)
xoutRgb.setStreamName("video")


lowerBlue = np.array([150,100,70])
upperBlue = np.array([200,170,110])

def main():

vesc.run(0.5, 0)
steerAngle = 0.5
throttle = 0

#cap = cv2.VideoCapture('trackvid.mp4')
with dai.Device(pipeline) as device:

video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
while True:

videoIn = video.get()
frame = videoIn.getCvFrame()
frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(frame_hsv, (0,50,20), (5,255,255))
mask2 = cv2.inRange(frame_hsv, (120,50,20), (180,255,255))
redMask = cv2.bitwise_or(mask1, mask2)
#redMask = cv2.inRange(frame_hsv, lowerRed, upperRed)

blueMask = cv2.inRange(frame_hsv, lowerBlue, upperBlue)
#frame = cv2.bitwise_and(frame, frame, mask=redMask)

lines = cv2.HoughLinesP(blueMask, 1, np.pi/180, 100, np.array([]), 100, 1)
contours, hierarchy = cv2.findContours(redMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, (0,255,0), 3)

ind = 0
max_len = 0
yVals = []
xVals = []

for i, contour in enumerate(contours):
if cv2.contourArea(contour) < 300:
#print(cv2.contourArea(contour))
continue
for point in contour:
xVals.append(point[0][0])
yVals.append(point[0][1])

finalPoints = []

if len(yVals) > 4:
ind = yVals.index(min(yVals))
finalPoints.append([xVals[ind], yVals[ind]])
yVals.pop(ind)
xVals.pop(ind)

ind = yVals.index(max(yVals))
finalPoints.append([xVals[ind], yVals[ind]])
yVals.pop(ind)
xVals.pop(ind)

angle = np.arctan((finalPoints[1][1] - finalPoints[0][1])/(finalPoints[1][0] - finalPoints[0][0]))

print('angle: ', angle * 180 / np.pi)
angle = (angle * 180 / np.pi)

print('line[0]: ', finalPoints[0][0])
print('frame[0]: ', frame.shape[1])
if angle < 60:
if finalPoints[0][0] > (frame.shape[1] // 2) - 200:
steerAngle = 0.7
elif finalPoints[0][0] < (frame.shape[1] // 2) + 200:
steerAngle = 0.4
elif abs(angle) < 60:
if angle < 0:
if(abs(angle) < 45):
steerAngle = 0.75
elif(abs(angle) < 50):
steerAngle = 0.65
elif(abs(angle) < 60):
steerAngle = 0.55
elif angle > 0:
if (angle < 45):
steerAngle = 0.25
elif (angle < 50):
steerAngle = 0.35
elif (angle < 65):
steerAngle = 0.45
#cv2.waitKey(0)
print('steering angle: ', steerAngle)
else:
steerAngle = 0.5
cv2.putText(frame, "angle: {}".format(steerAngle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.line(frame, (finalPoints[0][0], finalPoints[0][1]), (finalPoints[1][0], finalPoints[1][1]), (0,255,0), 2)

else:
#print('lines')
if lines is not None:
for line in lines:
# print(line)
line = line[0]
angle = np.arctan((line[3] - line[1])/(line[2] - line[0]))
print('angle: ', angle * 180 / np.pi)
angle = (angle * 180 / np.pi)

print('line[0]: ', line[0])
print('frame[0]: ', frame.shape[1])
if line[0] > (frame.shape[1] // 2) - 100:
steerAngle = 0.65
throttle = 0.5
elif line[0] < (frame.shape[1] // 2) + 100:
steerAngle = 0.35
throttle = 0.5
elif abs(angle) < 60:
if angle < 0:
if(abs(angle) < 45):
steerAngle = 0.75
elif(abs(angle) < 50):
steerAngle = 0.65
elif(abs(angle) < 60):
steerAngle = 0.55
elif angle > 0:
if (angle < 45):
steerAngle = 0.25
elif (angle < 50):
steerAngle = 0.35
elif (angle < 65):
steerAngle = 0.45
cv2.putText(frame, "angle: {}".format(steerAngle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

vesc.run(steerAngle, throttle)
cv2.imshow('frame', frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
break
elif cv2.waitKey(25) & 0xFF == ord('s'):
throttle = 0.2

if __name__ == '__main__':
main()
