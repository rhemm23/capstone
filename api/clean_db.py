from pymongo import MongoClient

client = MongoClient()
db = client.capstone

db.images.delete_many({})
db.image_sets.delete_many({})
db.weight_sets.delete_many({})
db.google_drive_tasks.delete_many({})
db.weight_train_tasks.delete_many({})
