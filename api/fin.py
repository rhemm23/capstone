from mediapipe.python.solutions import face_detection as mp_det
from pymongo import MongoClient
from PIL import Image, ImageDraw

import numpy as np

import math
import cv2
import os

client = MongoClient()
db = client.capstone

# ti = db.demo_images.find()
# for t in ti:
#   cnt = 0
#   for image in t['sub_images']:
#     if image['is_face']:
#       cnt += 1
#   print(cnt)
# exit()

SCALES = {
  300: 240,
  240: 180,
  180: 120,
  120: 60,
  60: 20
}

def ipgu(image):
  size = len(image)
  sub_images_bb = []
  sub_images = []
  while True:
    for y in range(0, size - 10, 10):
      for x in range(0, size - 10, 10):
        sub_image = []
        for i in range(20):
          for j in range(20):
            sub_image.append(image[y + i][x + j])
        rscl = 300 / size
        sub_images_bb.append([
          int(rscl * x),
          int(rscl * y),
          int(rscl * (x + 20)),
          int(rscl * (y + 20))
        ])
        sub_images.append(sub_image)
    if size in SCALES:
      scale = SCALES[size] / size
      next_img = [[0 for _ in range(SCALES[size])] for _ in range(SCALES[size])]
      next_img_cnts = [[0 for _ in range(SCALES[size])] for _ in range(SCALES[size])]
      for y in range(size):
        for x in range(size):
          ny = int(y * scale)
          nx = int(x * scale)
          next_img_cnts[ny][nx] += 1
          next_img[ny][nx] += image[y][x]
      for y in range(SCALES[size]):
        for x in range(SCALES[size]):
          next_img[y][x] = int(next_img[y][x] / next_img_cnts[y][x])
      image = next_img
      size = SCALES[size]
    else:
      break
  return sub_images, sub_images_bb

dir = '/home/rh/raw_images/'
image_paths = os.listdir(dir)

with mp_det.FaceDetection() as det:
  for image_path in image_paths:
    full_path = os.path.join(dir, image_path)
    img_data = None
    with open(full_path, 'rb') as file:
      img_data = np.frombuffer(file.read(), dtype=np.uint8).reshape((300, 300, 1))
    img = cv2.cvtColor(img_data, cv2.COLOR_GRAY2RGB)
    results = det.process(img).detections

    faces = []

    for result in results:
      face = result.location_data.relative_bounding_box
      reye = [
        int(result.location_data.relative_keypoints[0].x * 300),
        int(result.location_data.relative_keypoints[0].y * 300)
      ]
      leye = [
        int(result.location_data.relative_keypoints[1].x * 300),
        int(result.location_data.relative_keypoints[1].y * 300)
      ]
      x0 = int(face.xmin * 300)
      y0 = int(face.ymin * 300)
      x1 = x0 + int(face.width * 300)
      y1 = y0 + int(face.height * 300)
      ex = abs(reye[0] - leye[0])
      ey = abs(reye[1] - leye[1])
      rot = math.degrees(math.atan2(ey, ex))
      rot = rot if reye[1] < leye[1] else 360 - rot
      faces.append([[x0, y0, x1, y1], max(0, min(35, round(rot / 10))), 0, -1])

    sub_images, sub_images_bb = ipgu(img_data.reshape((300, 300)))

    for i in range(len(sub_images_bb)):
      for face in faces:
        bb = sub_images_bb[i]
        intersect = max(0, min(bb[2], face[0][2]) - max(bb[0], face[0][0])) * \
                    max(0, min(bb[3], face[0][3]) - max(bb[1], face[0][1]))
        union = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
                (face[0][2] - face[0][0]) * (face[0][3] - face[0][1]) - intersect
        ratio = intersect / union
        if ratio > face[2]:
          face[2] = ratio
          face[3] = i

    sub_image_docs = []
    for i in range(len(sub_images)):
      sub_face = None
      for face in faces:
        if (face[3] == i):
          sub_face = face
          break
      is_face = sub_face is not None
      sub_image_docs.append({
        'data': np.array(sub_images[i], dtype=np.uint8).tobytes(),
        'is_face': is_face,
        'rotation': sub_face[1] if is_face else -1
      })

    db.demo_images.insert_one({
      'sub_images': sub_image_docs,
      'data': img_data.reshape((90000,)).tobytes()
    })

    # scale = (300 / h) if (w > h) else (300 / w)

    # rh = int(h * scale)
    # rw = int(w * scale)

    # image = image.resize((rw, rh))
    # image = image.convert('L')

    # if rh > rw:
    #   pad = (rh - 300) // 2
    #   image = image.crop((0, pad, 300, pad + 300))
    # elif rw > rh:
    #   pad = (rw - 300) // 2
    #   image = image.crop((pad, 0, pad + 300, 300))

    # image_bytes = np.asarray(image).tobytes()
    # out_path = os.path.join('/home/rh/raw_images/', '{}.raw'.format(image_path.split('.')[0]))
    # with open(out_path, 'wb+') as file:
    #   file.write(image_bytes)
