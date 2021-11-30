from bson.objectid import ObjectId
from pymongo import MongoClient
from functools import wraps

import flask
import jwt
import os

client = MongoClient()
db = client.capstone

secret_key = os.urandom(24).hex()

def api_success(**kwargs):
  if kwargs:
    return flask.jsonify(kwargs), 200
  else:
    return '', 204

def api_error(error):
  return flask.jsonify({ 'error': error }), 400

def authenticate(route):
  @wraps(route)
  def decorated(*args, **kwargs):
    auth_header = flask.request.headers.get('authorization', None)
    if auth_header:
      parts = auth_header.split()
      if len(parts) == 2 and parts[0].lower() == 'bearer':
        try:
          token = jwt.decode(parts[1], secret_key, algorithms='HS256')
          account = db.accounts.find_one({ '_id': ObjectId(token['account_id']) })
          if account:
            return route(account, *args, **kwargs)
        except jwt.exceptions.InvalidTokenError:
          pass
    return api_error('Missing or invalid authorization token')
  return decorated
