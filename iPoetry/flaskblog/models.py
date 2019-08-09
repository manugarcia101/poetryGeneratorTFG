from datetime import datetime
from flaskblog import mongo
import re

def insertPoem(Poem):
    mongo.db.poemdb.insert(Poem)

def deletePoem():
    currDB = mongo.db.poemdb
    object_del = mongo.db.poemdb.find_one({"qty" : 15})
    currDB.remove(object_del)

def findPoem(Poem):
    array = []
    expr = re.compile(r''.join(Poem), re.I)
    for doc in mongo.db.poemdb.find({"Title": {'$regex': expr}}).sort("Fecha", -1).limit(30):
        array.append(doc)
    return array

def findAll():
    search = mongo.db.poemdb.find().sort("Fecha", -1).limit(30)
    array = []
    for doc in search:
        array.append(doc)
    return array
