from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from bson.objectid import ObjectId
from retinaface import RetinaFace
from pymongo import MongoClient
from celery import Celery
from io import BytesIO
from PIL import Image
from rnn import RNN
from dnn import DNN

import numpy as np
import math

cel = Celery(
  'tasks',
  backend='redis://localhost',
  broker='redis://localhost'
)

@cel.task
def weight_train_task(weight_train_task_id, image_set_id, weight_set_id):
  weight_train_task_id = ObjectId(weight_train_task_id)
  weight_set_id = ObjectId(weight_set_id)
  image_set_id = ObjectId(image_set_id)
  with MongoClient() as client:
    db = client.capstone
    weight_train_task = db.weight_train_tasks.find_one_and_update(
      {
        '_id': weight_train_task_id
      },
      {
        '$set': {
          'status': 'RUNNING'
        }
      }
    )

    weight_set = db.weight_sets.find_one({ '_id': weight_set_id })
    rot_nn = RNN(weight_set['weights']['rnn'])
    det_nn = DNN(weight_set['weights']['dnn'])

    for image in weight_train_task['images']:
      db.weight_train_tasks.update_one(
        {
          '_id': weight_train_task_id,
          'images.id': image['id']
        },
        {
          '$set': {
            'images.$.status': 'RUNNING'
          }
        }
      )
      image_doc = db.images.find_one(
        { '_id': image['id'] },
        { 'data': 0 }
      )

      # Ten epochs
      for _ in range(10):

        # Train on faces
        for face in image_doc['faces']:
          det_in = np.frombuffer(face, dtype=np.uint8).tolist()
          det_nn.train(det_in, 1)
        
        # Train on non-faces
        for non_face in image_doc['non_faces']:
          det_in = np.frombuffer(non_face, dtype=np.uint8).tolist()
          det_nn.train(det_in, 0)

        # Train on rotated samples
        for sample in image_doc['rotated_samples']:
          for i in range(36):
            rot_in = np.frombuffer(sample[i * 10], dtype=np.uint8).tolist()
            expected_output = [0 for _ in range(36)]
            expected_output[i] = 1
            rot_nn.train(rot_in, expected_output)

      db.weight_train_tasks.update_one(
        {
          '_id': weight_train_task_id,
          'images.id': image['id']
        },
        {
          '$set': {
            'images.$.status': 'DONE'
          }
        }
      )

    updated_weights = {
      'rnn': rot_nn.dict(),
      'dnn': det_nn.dict()
    }
    db.weight_sets.update_one(
      { '_id': weight_set_id },
      {
        '$set': {
          'weights': updated_weights
        }
      }
    )
    weight_train_task = db.weight_train_tasks.find_one_and_update(
      {
        '_id': weight_train_task_id
      },
      {
        '$set': {
          'status': 'DONE'
        }
      }
    )

@cel.task
def google_drive_task(google_drive_task_id, image_set_id, google_creds):
  google_drive_task_id = ObjectId(google_drive_task_id)
  google_creds = Credentials(**google_creds)
  image_set_id = ObjectId(image_set_id)
  service = build('drive', 'v3', credentials=google_creds)
  with MongoClient() as client:
    db = client.capstone
    google_drive_task = db.google_drive_tasks.find_one_and_update(
      {
        '_id': google_drive_task_id
      },
      {
        '$set': {
          'status': 'RUNNING'
        }
      }
    )
    for file in google_drive_task['files']:
      db.google_drive_tasks.update_one(
        {
          '_id': google_drive_task_id,
          'files.google_file_id': file['google_file_id']
        },
        {
          '$set': {
            'files.$.status': 'RUNNING'
          }
        }
      )
      request = service.files().get_media(fileId=file['google_file_id'])
      file_data = BytesIO()
      downloader = MediaIoBaseDownload(file_data, request, chunksize=1048576)
      done = False
      success = True
      while done is False:
        try:
          _, done = downloader.next_chunk()
        except HttpError:
          db.google_drive_tasks.update_one(
            {
              '_id': google_drive_task_id,
              'files.google_file_id': file['google_file_id']
            },
            {
              '$set': {
                'files.$.status': 'FAILED'
              }
            }
          )
          done = True
          success = False
    
      if not success:
        continue

      image = Image.open(file_data)
      width, height = image.size

      results = RetinaFace.detect_faces(np.asarray(image))
      faces = list(results.values()) if type(results) is dict else []

      scale = 300 / height if width > height else 300 / width

      rw = int(width * scale)
      rh = int(height * scale)

      cropped = image.resize((rw, rh))
      cropped = cropped.convert('L')

      rotated_samples = []
      face_sub_images = []
      face_bboxes = []

      for face in faces:

        x0 = int(scale * face['facial_area'][0])
        y0 = int(scale * face['facial_area'][1])
        x1 = int(scale * face['facial_area'][2])
        y1 = int(scale * face['facial_area'][3])

        dx = x1 - x0
        dy = y1 - y0
        d = (dx + dy) / 2

        sizes = [
          20,
          25,
          33,
          50,
          100,
          300
        ]

        fsize = min(sizes, key=lambda size: abs(size - d))
        padding = math.ceil((math.sqrt(2) - 1) * (fsize / 2))

        cx = int(x0 + (dx / 2))
        cy = int(y0 + (dy / 2))

        x0 = cx - (fsize // 2)
        y0 = cy - (fsize // 2)
        x1 = cx + (fsize // 2)
        y1 = cy + (fsize // 2)

        lb = x0 - padding
        rb = x1 + padding
        tb = y0 - padding
        bb = y1 + padding

        if lb < 0 or tb < 0 or rb > rw or bb > rh:
          continue

        face_bboxes.append([x0, y0, x1, y1])

        rex = int(scale * face['landmarks']['right_eye'][0])
        rey = int(scale * face['landmarks']['right_eye'][1])
        lex = int(scale * face['landmarks']['left_eye'][0])
        ley = int(scale * face['landmarks']['left_eye'][1])

        ex = abs(rex - lex)
        ey = abs(rey - ley)

        base_rotation = math.degrees(math.atan2(ey, ex))
        if rey > ley:
          base_rotation = 360 - base_rotation

        face_rot_samples = []
        for rotation in range(360):
          face_image = cropped.rotate(
            base_rotation - rotation,
            resample=Image.BICUBIC,
            center=(cx, cy)
          )
          face_image = face_image.crop((x0, y0, x1, y1))
          face_image = face_image.resize((20, 20))

          # Store aligned copy
          if rotation == 0:
            face_sub_images.append(np.asarray(face_image).tobytes())

          # Store sample
          face_rot_samples.append(np.asarray(face_image).tobytes())

        rotated_samples.append(face_rot_samples)

      temp_cropped = cropped.crop((0, 0, 300, 300))
      non_face_sub_images = []

      # Search for sub images in initial 300x300 not in a facial region
      for y in range(0, 300, 20):
        for x in range(0, 300, 20):
          valid = True
          for bbox in face_bboxes:
            x0, y0, x1, y1 = bbox
            if (x >= x0 and x < x1) or (x + 20 >= x0 and x + 20 < x1) or (y >= y0 and y < y1) or (y + 20 >= y0 and y + 20 < y1):
              valid = False
              break
          if valid:
            sub_image = temp_cropped.crop((x, y, x + 20, y + 20))
            non_face_sub_images.append(np.asarray(sub_image).tobytes())

      db.images.insert_one({
        'name': file['name'],
        'width': rw,
        'height': rh,
        'data': np.asarray(cropped).tobytes(),
        'faces': face_sub_images,
        'image_set_id': image_set_id,
        'non_faces': non_face_sub_images,
        'rotated_samples': rotated_samples
      })
      db.google_drive_tasks.update_one(
        {
          '_id': google_drive_task_id,
          'files.google_file_id': file['google_file_id']
        },
        {
          '$set': {
            'files.$.status': 'SUCCESS'
          }
        }
      )