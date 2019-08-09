from flask import Flask
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['MONGO_URI'] = "mongodb://localhost:27017/poemdb"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

from flaskblog import routes
