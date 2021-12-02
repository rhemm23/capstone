from google.oauth2.credentials import Credentials
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from bson.objectid import ObjectId
from bson.errors import InvalidId
from common import *

import tasks
import flask

google_drive_tasks = flask.Blueprint('google_drive_tasks', __name__)

@google_drive_tasks.route('/api/google-drive-tasks/<id>', methods=['GET'])
@authenticate
def get_google_drive_task(account, id):
  try:
    id = ObjectId(id)
  except InvalidId:
    return api_error('Invalid id')
  google_drive_task = db.google_drive_tasks.find_one({ '_id': id, 'account_id': account['_id'] })
  if not google_drive_task:
    return api_error('Google drive task does not exist')
  files = []
  for file in google_drive_task['files']:
    files.append({
      'name': file['name'],
      'status': file['status']
    })
  return api_success(
    id=str(id),
    files=files,
    status=google_drive_task['status']
  )

@google_drive_tasks.route('/api/google-drive-tasks', methods=['GET'])
@authenticate
def get_google_drive_tasks(account):
  pipeline = [
    {
      '$match': {
        'account_id': account['_id']
      }
    },
    {
      '$project': {
        '_id': 0,
        'id': {
          '$toString': '$_id'
        },
        'status': 1,
        'percent_complete': {
          '$toInt': {
            '$multiply': [
              {
                '$divide': [
                  {
                    '$reduce': {
                      'input': '$files',
                      'initialValue': 0,
                      'in': {
                        '$add': [
                          '$$value',
                          {
                            '$cond': {
                              'if': {
                                '$in': [
                                  '$$this.status',
                                  [
                                    'FAILED',
                                    'SUCCESS'
                                  ]
                                ]
                              },
                              'then': 1,
                              'else': 0
                            }
                          }
                        ]
                      }
                    }
                  },
                  {
                    '$size': '$files'
                  }
                ]
              },
              100
            ]
          }
        }
      }
    }
  ]
  return api_success(
    google_drive_tasks=list(db.google_drive_tasks.aggregate(pipeline))
  )

@google_drive_tasks.route('/api/google-drive-tasks', methods=['POST'])
@authenticate
def start_google_drive_task(account):
  image_set_id = flask.request.form.get('image_set_id', None)
  google_file_id = flask.request.form.get('google_file_id', None)
  if not image_set_id or not google_file_id:
    return api_error('Missing image set id or google file id')
  try:
    image_set_id = ObjectId(image_set_id)
  except InvalidId:
    return api_error('Invalid image set id')
  if db.image_sets.count_documents({ 'account_id': account['_id'], '_id': image_set_id }, limit=1) == 0:
    return api_error('Image set does not exist')
  google_creds = db.google_credentials.find_one(
    {
      'account_id': account['_id']
    },
    {
      '_id': 0,
      'email': 0,
      'account_id': 0
    }
  )
  if not google_creds:
    return api_error('No google credentials are associated with this account')
  credentials = Credentials(**google_creds)
  google_files = []
  try:
    service = build('drive', 'v3', credentials=credentials)
    file = service.files().get(fileId=google_file_id).execute()
    if file['mimeType'] in ['image/jpeg', 'image/png']:
      google_files.append({
        'id': google_file_id,
        'name': file['name']
      })
    elif file['mimeType'] == 'application/vnd.google-apps.folder':
      page_token = None
      while True:
        response = service.files().list(
          q='("{}" in parents) and (mimeType = "image/png" or mimeType = "image/jpeg")'.format(google_file_id),
          fields='nextPageToken, files(id, name)',
          pageToken=page_token
        ).execute()
        if response['files']:
          google_files += response['files']
        page_token = response.get('nextPageToken', None)
        if not page_token:
          break
      if len(google_files) == 0:
        return api_error('Folder does not contain any images')
  except HttpError:
    return api_error('Invalid google file id')
  files = []
  for google_file in google_files:
    files.append({
      'google_file_id': google_file['id'],
      'name': google_file['name'],
      'status': 'PENDING'
    })
  google_drive_task = {
    'image_set_id': image_set_id,
    'account_id': account['_id'],
    'status': 'PENDING',
    'files': files
  }
  google_drive_task_id = db.google_drive_tasks.insert_one(google_drive_task).inserted_id
  task = tasks.google_drive_task.delay(
    str(google_drive_task_id),
    str(image_set_id),
    google_creds
  )
  db.google_drive_tasks.update_one(
    {
      '_id': google_drive_task_id
    },
    {
      '$set': {
        'task_id': task.id
      }
    }
  )
  return api_success(id=str(google_drive_task_id))
