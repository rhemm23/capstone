from bson.objectid import ObjectId
from bson.errors import InvalidId

from common import *
from rnn import RNN
from dnn import DNN

import numpy as np
import tasks
import flask
import math

weights = flask.Blueprint('weights', __name__)

def generate_new_weight(fan_in, fan_out):
  dist = math.sqrt(6 / (fan_in + fan_out))
  weights = np.random.uniform(low=-dist, high=dist, size=(fan_in,))
  return {
    'bias': 0,
    'weights': weights.tolist()
  }

@weights.route('/api/weight-sets/<id>/test', methods=['POST'])
@authenticate
def test_weights(account, id):
  image_set_id = flask.request.form.get('image_set_id', None)
  if not image_set_id:
    return api_error('Missing image set id')
  try:
    id = ObjectId(id)
    image_set_id = ObjectId(image_set_id)
  except InvalidId:
    return api_error('Invalid id or image set id')
  weight_set = db.weight_sets.find_one({ 'account_id': account['_id'], '_id': id })
  if not weight_set:
    return api_error('Weight set does not exist')
  if db.image_sets.count_documents({ 'account_id': account['_id'], '_id': image_set_id }, limit=1) == 0:
    return api_error('Image set does not exist')

  results = db.images.find({ 'image_set_id': image_set_id })

  rnn = RNN(weight_set['weights']['rnn'])
  dnn = DNN(weight_set['weights']['dnn'])

  dnn_tot_cnt = 0
  dnn_pass_cnt = 0

  rnn_tot_cnt = 0
  rnn_pass_cnt = 0

  for image in results:

    # Test faces
    for face in image['faces']:
      dnn_in = np.frombuffer(face, dtype=np.uint8).tolist()
      dnn_tot_cnt += 1
      dnn_pass_cnt += 1 if dnn.forward(dnn_in) else 0

    # Test non faces
    for non_face in image['non_faces']:
      dnn_in = np.frombuffer(non_face, dtype=np.uint8).tolist()
      dnn_tot_cnt += 1
      dnn_pass_cnt += 1 if not dnn.forward(dnn_in) else 0

    # Test rotation samples
    for sample in image['rotated_samples']:
      for i in range(36):
        rot_in = np.frombuffer(sample[i * 10], dtype=np.uint8).tolist()
        expected = [0 for _ in range(36)]
        actual = rnn.forward(rot_in)
        expected[i] = 1
        rnn_tot_cnt += 1
        eq = True
        for j in range(36):
          if expected[j] != actual[i]:
            eq = False
            break
        rnn_pass_cnt += 1 if eq else 0

  return api_success(
    rnn_accuracy=round(dnn_pass_cnt / dnn_tot_cnt, 2),
    dnn_accuracy=round(rnn_pass_cnt / rnn_tot_cnt, 2)
  )


@weights.route('/api/weight-sets/<id>/train', methods=['POST'])
@authenticate
def train_weights(account, id):
  image_set_id = flask.request.form.get('image_set_id', None)
  if not image_set_id:
    return api_error('Missing image set id')
  try:
    id = ObjectId(id)
    image_set_id = ObjectId(image_set_id)
  except InvalidId:
    return api_error('Invalid id or image set id')
  if db.weight_sets.count_documents({ 'account_id': account['_id'], '_id': id }, limit=1) == 0:
    return api_error('Weight set does not exist')
  if db.image_sets.count_documents({ 'account_id': account['_id'], '_id': image_set_id }, limit=1) == 0:
    return api_error('Image set does not exist')
  results = db.images.find({ 'image_set_id': image_set_id }, projection={ 'name': 1 })
  images = []
  for image in results:
    images.append({
      'id': image['_id'],
      'name': image['name'],
      'status': 'PENDING'
    })
  weight_train_task = {
    'images': images,
    'status': 'PENDING',
    'weight_set_id': id,
    'account_id': account['_id'],
    'image_set_id': image_set_id
  }
  weight_train_task_id = db.weight_train_tasks.insert_one(weight_train_task).inserted_id
  task = tasks.weight_train_task.delay(
    str(weight_train_task_id),
    str(image_set_id),
    str(id)
  )
  db.weight_train_tasks.update_one(
    { '_id': weight_train_task_id },
    {
      '$set': {
        'task_id': task.id
      }
    }
  )
  return api_success(id=str(weight_train_task_id))

@weights.route('/api/weight-sets/<id>', methods=['POST'])
@authenticate
def update_weight_set(account, id):
  name = flask.request.form.get('name', None)
  if not name:
    return api_error('Missing name')
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  if db.weight_sets.count_documents({ 'account_id': account['_id'], 'name': name }, limit=1) == 1:
    return api_error('Name already in use')
  if db.weight_sets.update_one({ 'account_id': account['_id'], '_id': id }, { '$set': { 'name': name } }).matched_count == 0:
    return api_error('Weight set does not exist')
  return api_success()

@weights.route('/api/weight-sets/<id>', methods=['GET'])
@authenticate
def get_weight_set(account, id):
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  weight_set = db.weight_sets.find_one({ 'account_id': account['_id'], '_id': id }, projection={ 'name': 1, 'weights': 1 })
  if not weight_set:
    return api_error('Weight set does not exist')
  return api_success(id=str(weight_set['_id']), name=weight_set['name'], weights=weight_set['weights'])

@weights.route('/api/weight-sets/<id>', methods=['DELETE'])
@authenticate
def delete_weight_set(account, id):
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  if db.weight_sets.delete_one({ 'account_id': account['_id'], '_id': id }).deleted_count == 1:
    return api_success()
  else:
    return api_error('Weight set does not exist')

@weights.route('/api/weight-sets', methods=['GET'])
@authenticate
def get_weights(account):
  weight_sets = db.weight_sets.find({ 'account_id': account['_id'] }, projection={ 'name': 1 })
  results = []
  for weight_set in weight_sets:
    results.append({
      'name': weight_set['name'],
      'id': str(weight_set['_id'])
    })
  return api_success(weight_sets=results)

@weights.route('/api/weight-sets', methods=['POST'])
@authenticate
def create_weight_set(account):
  name = flask.request.form.get('name', None)
  if not name:
    return api_error('Missing name')
  if db.weight_sets.count_documents({ 'account_id': account['_id'], 'name': name }, limit=1) == 1:
    return api_error('Name already in use')
  weight_set = {
    'account_id': account['_id'],
    'name': name,
    'weights': {
      'rnn': {
        'a_layer': [generate_new_weight(400, 15) for _ in range(15)],
        'b_layer': [generate_new_weight(15, 15) for _ in range(15)],
        'c_layer': [generate_new_weight(15, 1) for _ in range(36)]
      },
      'dnn': {
        'a_layer': {
          'type1': [[generate_new_weight(100, 1) for _ in range(2)] for _ in range(2)],
          'type2': [[generate_new_weight(25, 1) for _ in range(4)] for _ in range(4)],
          'type3': [generate_new_weight(80, 1) for _ in range(5)]
        },
        'b_type1': generate_new_weight(4, 1),
        'b_type2': generate_new_weight(16, 1),
        'b_type3': generate_new_weight(5, 1),
        'c': generate_new_weight(3, 1)
      }
    }
  }
  weight_set_id = db.weight_sets.insert_one(weight_set)
  return api_success(id=str(weight_set_id.inserted_id))
