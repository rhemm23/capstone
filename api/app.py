from common import secret_key
from account import account
from weights import weights
from images import images

import flask

app = flask.Flask(__name__)
app.secret_key = secret_key

app.register_blueprint(account)
app.register_blueprint(weights)
app.register_blueprint(images)

if __name__ == '__main__':
  app.run()
