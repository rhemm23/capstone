from PIL import Image, ImageDraw
from pymongo import MongoClient
from weights import *
from ipgu import ipgu
from heu import heu
from rnn import *
from dnn import *

import getopt
import math
import dlib
import sys

RNN_GAMMA = 0.05
RNN_BIAS_GAMMA = 0.01

DNN_GAMMA = 0.05
DNN_BIAS_GAMMA = 0.01

def dtanh(value):
  return 1 - math.tanh(value)**2

def train_rnn(sub_image, expected_output, rnw):
  a_out, b_out, c_out = rnn_layer_out(sub_image, rnw)
  c_err = [(c_out[i] - expected_output[i]) * dtanh(c_out[i]) for i in range(36)]
  b_err = []
  a_err = []
  for i in range(15):
    err = 0
    for j in range(36):
      err += c_err[j] * rnw[2][j][1][i]
    err *= dtanh(b_out[i])
    b_err.append(err)
  for i in range(15):
    err = 0
    for j in range(15):
      err += b_err[j] * rnw[1][j][1][i]
    err *= dtanh(a_out[i])
    a_err.append(err)
  for i in range(15):
    for j in range(400):
      rnw[0][i][1][j] -= RNN_GAMMA * a_err[i] * sub_image[j]
    rnw[0][i] = (rnw[0][i][0] - RNN_BIAS_GAMMA * a_err[i], rnw[0][i][1])
  for i in range(15):
    for j in range(15):
      rnw[1][i][1][j] -= RNN_GAMMA * b_err[i] * a_out[j]
    rnw[1][i] = (rnw[1][i][0] - RNN_BIAS_GAMMA * b_err[i], rnw[1][i][1])
  for i in range(36):
    for j in range(15):
      rnw[2][i][1][j] -= RNN_GAMMA * c_err[i] * b_out[j]
    rnw[2][i] = (rnw[2][i][0] - RNN_BIAS_GAMMA * c_err[i], rnw[2][i][1])

def train_dnn(sub_image, expected_output, dnw):
  a1_in, a2_in, a3_in = arrange_input(sub_image)
  a_out, b_out, c_out = dnn_layer_out(sub_image, dnw)
  c_err = (c_out - expected_output) * dtanh(c_out)
  b_err = [c_err * dnw[4][1][i] * dtanh(b_out[i]) for i in range(3)]
  a1_err = [b_err[0] * dnw[3][0][1][i] * dtanh(a_out[0][i]) for i in range(4)]
  a2_err = [b_err[1] * dnw[3][1][1][i] * dtanh(a_out[1][i]) for i in range(16)]
  a3_err = [b_err[2] * dnw[3][2][1][i] * dtanh(a_out[2][i]) for i in range(5)]
  for i in range(4):
    for j in range(100):
      dnw[0][i][1][j] -= DNN_GAMMA * a1_err[i] * a1_in[i][j]
    dnw[0][i] = (dnw[0][i][0] - DNN_BIAS_GAMMA * a1_err[i], dnw[0][i][1])
  for i in range(16):
    for j in range(25):
      dnw[1][i][1][j] -= DNN_GAMMA * a2_err[i] * a2_in[i][j]
    dnw[1][i] = (dnw[1][i][0] - DNN_BIAS_GAMMA * a2_err[i], dnw[1][i][1])
  for i in range(5):
    for j in range(80):
      dnw[2][i][1][j] -= DNN_GAMMA * a3_err[i] * a3_in[i][j]
    dnw[2][i] = (dnw[2][i][0] - DNN_BIAS_GAMMA * a3_err[i], dnw[2][i][1])
  for i in range(4):
    dnw[3][0][1][i] -= DNN_GAMMA * b_err[0] * a_out[0][i]
  dnw[3][0] = (dnw[3][0][0] - DNN_BIAS_GAMMA * b_err[0], dnw[3][0][1])
  for i in range(16):
    dnw[3][1][1][i] -= DNN_GAMMA * b_err[1] * a_out[1][i]
  dnw[3][1] = (dnw[3][1][0] - DNN_BIAS_GAMMA * b_err[1], dnw[3][1][1])
  for i in range(5):
    dnw[3][2][1][i] -= DNN_GAMMA * b_err[2] * a_out[2][i]
  dnw[3][2] = (dnw[3][2][0] - DNN_BIAS_GAMMA * b_err[2], dnw[3][2][1])
  for i in range(3):
    dnw[4][1][i] -= DNN_GAMMA * c_err * b_out[i]
  dnw[4] = (dnw[4][0] - DNN_BIAS_GAMMA * c_err, dnw[4][1])

def train_rnn_image(image, rnw):

  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')

  faces = detector(image)

  print('Detected {0} faces..'.format(len(faces)))

  face_dets = []
  for face in faces:
    landmarks = predictor(image=image, box=face)
    reye = [
      (landmarks.part(0).x + landmarks.part(1).x) / 2,
      (landmarks.part(0).y + landmarks.part(1).y) / 2
    ]
    leye = [
      (landmarks.part(2).x + landmarks.part(3).x) / 2,
      (landmarks.part(2).y + landmarks.part(3).y) / 2
    ]

    ex = abs(reye[0] - leye[0])
    ey = abs(reye[1] - leye[1])

    rot = math.degrees(math.atan2(ey, ex))
    if reye[1] < leye[1]:
      rot = 360 - rot
    
    face_dets.append([
      [
        face.left(),
        face.top(),
        face.right(),
        face.bottom()
      ],
      -1,
      None,
      rot
    ])

  image = image.tolist()
  sub_images, sub_images_bb = ipgu(image)
  face_sub_image_indices = []

  for i in range(len(sub_images_bb)):
    for face_det in face_dets:
      bb = sub_images_bb[i]
      intersect = max(0, min(bb[2], face_det[0][2]) - max(bb[0], face_det[0][0])) * \
                  max(0, min(bb[3], face_det[0][3]) - max(bb[1], face_det[0][1]))
      union = (bb[2] - bb[0]) * (bb[3] - bb[1]) + \
              (face_det[0][2] - face_det[0][0]) * (face_det[0][3] - face_det[0][1]) - intersect
      ratio = intersect / union
      if ratio > face_det[1]:
        face_det[1] = ratio
        face_det[2] = i
    
  for face_det in face_dets:
    face_sub_image_indices.append(face_det[2])

  for i in range(len(sub_images)):
    heu(sub_images[i])
    if i in face_sub_image_indices:
      rot = None
      for face_det in face_dets:
        if face_det[2] == i:
          rot = face_det[3]
          break
      expected = [0 for _ in range(36)]
      expected[round(rot / 10) % 36] = 1
      train_rnn(sub_images[i], expected, rnw)

if __name__ == '__main__':
  rnw_path = None
  dnw_path = None
  sys_args = sys.argv[1:]
  try:
    args, vals = getopt.getopt(sys_args, 'r:d:', ['rnw =', 'dnw ='])
    for arg, val in args:
      if arg in ('-r', '--rnw'):
        rnw_path = val
      elif arg in ('-d', '--dnw'):
        dnw_path = val
  except getopt.error as err:
    print(str(err))
    exit()
  if not rnw_path:
    print('Error: No rotational neural net weight specified')
    exit()

  rnw = read_rnn_weights(rnw_path) if rnw_path else None
  dnw = read_dnn_weights(dnw_path) if dnw_path else None

  client = MongoClient()
  db = client.capstone

  # Ten epochs
  for _ in range(10):

    cnt = 0
    for image in db.images.find():

      w = image['width']
      h = image['height']
      data = image['data']

      image = Image.frombuffer('L', (w, h), data)

      if w > h:
        pad = (w - 300) // 2
        image = image.crop((pad, 0, pad + 300, 300))
      elif h > w:
        pad = (h - 300) // 2
        image = image.crop((0, pad, 300, pad + 300))

      train_rnn_image(np.asarray(image), rnw)

      cnt += 1
      print('Completed image number: {}'.format(cnt))

      # Write weights every 100 images
      if cnt % 100 == 0:
        write_rnn_weights(rnw_path, rnw)
