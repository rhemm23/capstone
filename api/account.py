from common import *

import bcrypt
import flask
import jwt

account = flask.Blueprint('account', __name__)

@account.route('/api/token', methods=['POST'])
def new_token():
  username = flask.request.form.get('username', None)
  password = flask.request.form.get('password', None)
  if not username or not password:
    return api_error('Missing username or password')
  account = db.accounts.find_one({ 'username': username })
  if not account or not bcrypt.checkpw(str.encode(password), account['password_hash']):
    return api_error('Invalid username or account')
  return api_success(token=jwt.encode({ 'account_id': str(account['_id']) }, secret_key))

@account.route('/api/account', methods=['DELETE'])
@authenticate
def delete_account(account):
  db.accounts.delete_one({ '_id': account['_id'] })
  return api_success()

@account.route('/api/account', methods=['PUT'])
@authenticate
def update_account(account):
  password = flask.request.form.get('password', None)
  if not password:
    return api_error('Missing password')
  if bcrypt.checkpw(str.encode(password), account['password_hash']):
    return api_error('Password already in use')
  hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
  db.accounts.update_one({ '_id': account['_id'] }, { '$set': { 'password_hash': hash } })
  return api_success()

@account.route('/api/account', methods=['GET'])
@authenticate
def get_account(account):
  return api_success(id=str(account['_id']), username=account['username'])

@account.route('/api/account', methods=['POST'])
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
  return api_success(token=jwt.encode({ 'account_id': str(account_id) }, secret_key))
