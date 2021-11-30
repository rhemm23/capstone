from bson.objectid import ObjectId
from bson.errors import InvalidId

from common import *

import numpy as np
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
