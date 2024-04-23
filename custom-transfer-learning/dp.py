import pymongo
import json


# Connect to MongoDB - adjust the connection string as needed
client = pymongo.MongoClient("mongodb://localhost:27017/")

# Create a database called "home_automation"
db = client["object_detection"]

# Create a collection called "light_switches"
collection = db["items"]

# Load data from JSON file
with open("data.json", "r") as file:
    data = json.load(file)

# Insert data into the collection
collection.insert_many(data)

print("Data inserted successfully!")

# Optionally, retrieve and print all documents in the collection to confirm
for switch in collection.find():
    print(switch)
