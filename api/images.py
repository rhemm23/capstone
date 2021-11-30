from bson.objectid import ObjectId
from bson.errors import InvalidId

from common import *

import flask

images = flask.Blueprint('images', __name__)

@images.route('/api/image-sets/<id>', methods=['GET'])
@authenticate
def get_image_set(account, id):
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  image_set = db.image_sets.find_one({ 'account_id': account['_id'], '_id': id }, projection={ 'name': 1 })
  if not image_set:
    return api_error('Image set does not exist')
  results = db.images.find({ 'image_set_id': id }, projection={ 'name': 1 })
  images_res = []
  for image in results:
    images_res.append({
      'id': str(image['_id']),
      'name': image['name']
    })
  return api_success(id=str(image_set['_id']), name=image_set['name'], images=images_res)

@images.route('/api/image-sets', methods=['POST'])
@authenticate
def create_image_set(account):
  name = flask.request.form.get('name', None)
  if not name:
    return api_error('Missing name')
  if db.image_sets.count_documents({ 'account_id': account['_id'], 'name': name }, limit=1) == 1:
    return api_error('Name is already in use')
  result = db.image_sets.insert_one({ 'account_id': account['_id'], 'name': name })
  return api_success(id=str(result.inserted_id))

@images.route('/api/image-sets', methods=['GET'])
@authenticate
def get_image_sets(account):
  results = db.image_sets.find({ 'account_id': account['_id'] })
  image_sets = []
  for image_set in results:
    image_sets.append({
      'id': str(image_set['_id']),
      'name': image_set['name']
    })
  return api_success(image_sets=image_sets)

@images.route('/api/image-sets/<id>', methods=['DELETE'])
@authenticate
def delete_image_set(account, id):
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  if db.image_sets.delete_one({ 'account_id': account['_id'], '_id': id }).deleted_count == 1:
    return api_success()
  return api_error('Image set does not exist')

@images.route('/api/image-sets/<id>', methods=['POST'])
@authenticate
def update_image_set(account, id):
  name = flask.request.form.get('name', None)
  if not name:
    return api_error('Missing name')
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  if db.image_sets.count_documents({ 'account_id': account['_id'], 'name': name }, limit=1) == 1:
    return api_error('Name is already in use')
  if db.image_sets.update_one({ 'account_id': account['_id'], '_id': id }, { '$set': { 'name': name } }).matched_count == 0:
    return api_error('Image set does not exist')
  return api_success()
