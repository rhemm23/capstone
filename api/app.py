from bson.objectid import ObjectId
from pymongo import MongoClient
from functools import wraps

import datetime
import bcrypt
import flask
import jwt
import os

client = MongoClient()
db = client.capstone

app = flask.Flask(__name__)
app.secret_key = os.urandom(24).hex()

def authenticate(route):
  @wraps(route)
  def decorated(*args, **kwargs):
    auth_header = flask.request.headers.get('authorization', None)
    if auth_header:
      parts = auth_header.split()
      if len(parts) == 2 and parts[0].lower() == 'bearer':
        try:
          token = jwt.decode(parts[1], app.secret_key, algorithms='HS256')
          account_id = ObjectId(token['account_id'])
          if db.accounts.count_documents({ '_id': account_id }, limit=1) == 1:
            return route(account_id, *args, **kwargs)
        except jwt.exceptions.InvalidTokenError:
          pass
    return api_error('Missing or invalid authorization token')
  return decorated

def api_success(**kwargs):
  result = { 'success': True }
  for key, value in kwargs.items():
    result[key] = value
  return flask.jsonify(result), 200

def api_error(error):
  result = {
    'success': False,
    'error': error
  }
  return flask.jsonify(result), 400

@app.route('/api/test')
@authenticate
def test(id):
  return api_success(id=str(id))

@app.route('/api/accounts', methods=['POST'])
def create_account():
  username = flask.request.form.get('username', None)
  password = flask.request.form.get('password', None)
  if not username or not password:
    return api_error('Missing username or password')
  if db.accounts.count_documents({ 'username': username }, limit=1) != 0:
    return api_error('Username is already in use')
  password_hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
  account_doc = { 'username': username, 'password_hash': password_hash }
  account_id = db.accounts.insert_one(account_doc).inserted_id
  return api_success(token=jwt.encode({ 'account_id': str(account_id) }, app.secret_key))

if __name__ == '__main__':
  app.run(port=5000)
