
# For monitoring web camera and performing image minipulations
import cv2
# For performing array operations
import numpy as np
# For creating and removing directories
import os
import shutil
# For recognizing and performing actions on mouse presses
from pynput.mouse import Listener


root = input("Enter the directory to store the images: ")
if os.path.isdir(root):
  resp = ""
  while not resp in ["Y", "N"]:
    resp = input("This directory already exists. If you continue, the contents of the existing directory will be deleted. If you would still like to proceed, enter [Y]. Otherwise, enter [N]: ")
  if resp == "Y": 
    shutil.rmtree(root)
  else:
    exit()
os.mkdir(root)


def normalize(x):
  minn, maxx = x.min(), x.max()
  return (x - minn) / (maxx - minn)


def scan(image_size=(32, 32)):
  _, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  boxes = cascade.detectMultiScale(gray, 1.3, 10)
  if len(boxes) == 2:
    eyes = []
    for box in boxes:
      x, y, w, h = box
      eye = frame[y:y + h, x:x + w]
      eye = cv2.resize(eye, image_size)
      eye = normalize(eye)
      eye = eye[10:-10, 5:-5]
      eyes.append(eye)
    return (np.hstack(eyes) * 255).astype(np.uint8)
  else:
    return None




# 这个函数能在每次触发onclick事件调用
def on_click2(x, y, button, pressed):
  """
  Args:
    x: the x-coordinate of the mouse
    y: the y-coordinate of the mouse
    button: 1 or 0, depending on right-click or left-click
    pressed: 1 or 0, whether the mouse was pressed or released
  """
  if pressed:
    print (x, y)
# 这就能用来实现自动化

def on_click(x,y,button,pressed):
  if pressed:
    eyes = scan()
    if not eyes is None:
      filename = root+f"{x} {y} {button}.jpeg"
      cv2.imwrite(filename,eyes)

cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

with Listener(on_click = on_click) as listener:
  listener.join()