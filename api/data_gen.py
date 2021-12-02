from mediapipe.python.solutions import face_detection as mp_det
from pymongo import MongoClient
from PIL import Image
from heu import heu

import numpy as np

import signal
import math
import cv2
import sys
import os

client = MongoClient()
db = client.capstone

TRAIN_DIR = '/mnt/d/vggface2/data/vggface2_train/train'
SPEC_PATH = '/mnt/d/vggface2/data/train_list.txt'
SAVE_PATH = './loader.state'

images = []
completed = 0

def signal_handler(sig, frame):
  print('Saving progress...')
  if completed > 0:
    with open(SAVE_PATH, 'w+') as save:
      save.writelines([str(completed)])
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Read progress
if os.path.exists(SAVE_PATH):
  with open(SAVE_PATH, 'r') as save:
    completed = int(save.readline())

# Read spec
with open(SPEC_PATH, 'r') as spec:
  images = spec.read().splitlines()

for image in images[completed:]:
  image = os.path.join(TRAIN_DIR, image)
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  pil = Image.fromarray(img).convert('L')
  reye = None
  leye = None
  face = None
  with mp_det.FaceDetection() as det:
    results = det.process(img).detections
    if not results:
      print('FAIL {}  /  {}'.format(completed + 1, len(images)))
      continue
    else:
      face = results[0].location_data.relative_bounding_box
      reye = [
        int(results[0].location_data.relative_keypoints[0].x * 128),
        int(results[0].location_data.relative_keypoints[0].y * 128)
      ]
      leye = [
        int(results[0].location_data.relative_keypoints[1].x * 128),
        int(results[0].location_data.relative_keypoints[1].y * 128)
      ]
      print('PROC {}  /  {}'.format(completed + 1, len(images)))

  x0 = int(face.xmin * 128)
  y0 = int(face.ymin * 128)
  x1 = x0 + int(face.width * 128)
  y1 = y0 + int(face.height * 128)
  dx = x1 - x0
  dy = y1 - y0
  cx = int(x0 + (dx / 2))
  cy = int(y0 + (dy / 2))
  dx = 2 * min(dx / 2, (128 - cx) / math.sqrt(2), cx / math.sqrt(2))
  dy = 2 * min(dy / 2, (128 - cy) / math.sqrt(2), cy / math.sqrt(2))
  d = min(dx, dy)
  ex = abs(reye[0] - leye[0])
  ey = abs(reye[1] - leye[1])
  rot = math.degrees(math.atan2(ey, ex))
  pil = pil.rotate(
    rot if reye[1] > leye[1] else 360 - rot,
    resample=Image.BICUBIC,
    center=(cx, cy)
  )
  for i in range(36):
    t = pil
    if i > 0:
      t = t.rotate(
        360 - (i * 10),
        resample=Image.BICUBIC,
        center=(cx, cy)
      )
    t = t.crop((cx - (d / 2), cy - (d / 2), cx + (d / 2), cy + (d / 2)))
    t = t.resize((20, 20))
    dat = np.asarray(t).reshape((400,)).tolist()
    heu(dat)
    samp = {
      'data': np.array(dat, dtype=np.uint8).tobytes(),
      'rotation': i * 10
    }
    db.rot_data.insert_one(samp)
  completed += 1
