from google_credentials import google_credentials
from google_drive_tasks import google_drive_tasks
from common import secret_key
from account import account
from weights import weights
from images import images

import flask
import os

app = flask.Flask(__name__)
app.secret_key = secret_key

app.register_blueprint(google_credentials)
app.register_blueprint(google_drive_tasks)
app.register_blueprint(account)
app.register_blueprint(weights)
app.register_blueprint(images)

if __name__ == '__main__':

  os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

  app.run()
