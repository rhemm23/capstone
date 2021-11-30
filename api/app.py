from pymongo import MongoClient
from account import account

import flask
import os

client = MongoClient()
db = client.capstone

app = flask.Flask(__name__)
app.secret_key = os.urandom(24).hex()

app.register_blueprint(account)

def api_success(**kwargs):
  if kwargs:
    return flask.jsonify(kwargs), 200
  else:
    return '', 204

def api_error(error):
  return flask.jsonify({ 'error': error }), 400

if __name__ == '__main__':
  app.run()
