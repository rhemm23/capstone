from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

from common import *

import flask

CLIENT_SECRETS_FILE = './credentials.json'

SCOPES = [
  'openid',
  'https://www.googleapis.com/auth/drive',
  'https://www.googleapis.com/auth/userinfo.email',
  'https://www.googleapis.com/auth/userinfo.profile'
]

google_credentials = flask.Blueprint('google_credentials', __name__)

@google_credentials.route('/api/google-credentials', methods=['GET'])
@authenticate
def get_google_credentials(account):
  results = db.google_credentials.find({ 'account_id': account['_id'] }, projection={ 'email': 1 })
  credentials = []
  for credential in results:
    credentials.append({
      'id': str(credential['_id']),
      'email': credential['email']
    })
  return api_success(google_credentials=credentials)

@google_credentials.route('/api/google-credentials/auth/callback', methods=['GET'])
def google_credential_auth_callback():
  state = flask.request.args.get('state', None)
  if not state:
    return api_error('Invalid state')
  state_doc = db.google_auth_states.find_one({ 'state': state })
  if not state_doc:
    return api_error('Invalid state')

  flow = Flow.from_client_secrets_file(
    CLIENT_SECRETS_FILE,
    scopes=SCOPES,
    state=state
  )
  flow.redirect_uri = flask.url_for('google_credentials.google_credential_auth_callback', _external=True)
  flow.fetch_token(authorization_response=flask.request.url)

  credentials = flow.credentials
  user_info_service = build('oauth2', 'v2', credentials=credentials)
  user_info = user_info_service.userinfo().get().execute()

  google_creds = {
    'email': user_info['email'],
    'token': credentials.token,
    'scopes': credentials.scopes,
    'token_uri': credentials.token_uri,
    'client_id': credentials.client_id,
    'account_id': state_doc['account_id'],
    'refresh_token': credentials.refresh_token,
    'client_secret': credentials.client_secret
  }

  result = db.google_credentials.insert_one(google_creds)
  return api_success(id=str(result.inserted_id))

@google_credentials.route('/api/google-credentials/auth', methods=['GET'])
@authenticate
def start_google_credential_auth(account):
  flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
  flow.redirect_uri = flask.url_for('google_credentials.google_credential_auth_callback', _external=True)
  authorization_url, state = flow.authorization_url(
    access_type='offline',
    include_granted_scopes='true'
  )
  db.google_auth_states.insert_one({
    'account_id': account['_id'],
    'state': state,
  })
  return api_success(url=authorization_url)
