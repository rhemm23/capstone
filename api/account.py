from bson.objectid import ObjectId
from functools import wraps

import bcrypt
import flask
import jwt

account = flask.Blueprint('account', __name__)

import app

def authenticate(route):
  @wraps(route)
  def decorated(*args, **kwargs):
    auth_header = flask.request.headers.get('authorization', None)
    if auth_header:
      parts = auth_header.split()
      if len(parts) == 2 and parts[0].lower() == 'bearer':
        try:
          token = jwt.decode(parts[1], app.app.secret_key, algorithms='HS256')
          account = app.db.accounts.find_one({ '_id': ObjectId(token['account_id']) })
          if account:
            return route(account, *args, **kwargs)
        except jwt.exceptions.InvalidTokenError:
          pass
    return app.api_error('Missing or invalid authorization token')
  return decorated

@account.route('/api/token', methods=['POST'])
def new_token():
  username = flask.request.form.get('username', None)
  password = flask.request.form.get('password', None)
  if not username or not password:
    return app.api_error('Missing username or password')
  account = app.db.accounts.find_one({ 'username': username })
  if not account or not bcrypt.checkpw(str.encode(password), account['password_hash']):
    return app.api_error('Invalid username or account')
  return app.api_success(token=jwt.encode({ 'account_id': str(account['_id']) }, app.app.secret_key))

@account.route('/api/account', methods=['DELETE'])
@authenticate
def delete_account(account):
  app.db.accounts.delete_one({ '_id': account['_id'] })
  return app.api_success()

@account.route('/api/account', methods=['PUT'])
@authenticate
def update_account(account):
  password = flask.request.form.get('password', None)
  if not password:
    return app.api_error('Missing password')
  if bcrypt.checkpw(str.encode(password), account['password_hash']):
    return app.api_error('Password already in use')
  hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
  app.db.accounts.update_one({ '_id': account['_id'] }, { '$set': { 'password_hash': hash } })
  return app.api_success()

@account.route('/api/account', methods=['GET'])
@authenticate
def get_account(account):
  return app.api_success(id=str(account['_id']), username=account['username'])

@account.route('/api/account', methods=['POST'])
def create_account():
  username = flask.request.form.get('username', None)
  password = flask.request.form.get('password', None)
  if not username or not password:
    return app.api_error('Missing username or password')
  if app.db.accounts.count_documents({ 'username': username }, limit=1) != 0:
    return app.api_error('Username is already in use')
  password_hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
  account_doc = { 'username': username, 'password_hash': password_hash }
  account_id = app.db.accounts.insert_one(account_doc).inserted_id
  return app.api_success(token=jwt.encode({ 'account_id': str(account_id) }, app.app.secret_key))
