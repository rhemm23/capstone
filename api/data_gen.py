from mediapipe.python.solutions import face_detection as mp_det
from pymongo import MongoClient
from PIL import Image

import numpy as np

import math
import cv2
import os

images = []
img_count = 0

for image in os.listdir('/home/rh/images'):
  images.append(os.path.join('/home/rh/images', image))

client = MongoClient()
db = client.capstone

with mp_det.FaceDetection() as det:
  for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img).convert('L')
    w, h = pil.size
    results = det.process(img).detections
    for result in results:
      face = result.location_data.relative_bounding_box
      reye = [
        int(result.location_data.relative_keypoints[0].x * w),
        int(result.location_data.relative_keypoints[0].y * h)
      ]
      leye = [
        int(result.location_data.relative_keypoints[1].x * w),
        int(result.location_data.relative_keypoints[1].y * h)
      ]
      x0 = int(face.xmin * w)
      y0 = int(face.ymin * h)
      x1 = x0 + int(face.width * w)
      y1 = y0 + int(face.height * h)
      dx = x1 - x0
      dy = y1 - y0
      cx = int(x0 + (dx / 2))
      cy = int(y0 + (dy / 2))
      dx = 2 * min(dx / 2, (w - cx) / math.sqrt(2), cx / math.sqrt(2))
      dy = 2 * min(dy / 2, (h - cy) / math.sqrt(2), cy / math.sqrt(2))
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
        cnt = [0 for _ in range(256)]
        for pixel in dat:
          cnt[pixel] += 1
        cdf = [cnt[0]]
        for j in range(1, 256):
          cdf.append(cdf[-1] + cnt[j])
        for j in range(400):
          dat[j] = int(cdf[dat[j]] * 256 / 400)
        samp = {
          'data': np.array(dat, dtype=np.uint8).tobytes(),
          'rotation': i * 10
        }
        db.rot_data.insert_one(samp)
      img_count += 1
      print(img_count)
