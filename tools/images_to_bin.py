from pymongo import MongoClient

import bitstring

client = MongoClient()
db = client.capstone

bin_data = bytes()

for image in db.demo_images.find().limit(30):
  bin_data += image['data']
  bin_data += bytes([0 for _ in range(48)])

with open('images.bin', 'wb+') as file:
  file.write(bin_data)
