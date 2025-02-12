import csv
from pymongo import MongoClient

# MongoDB connection
client = MongoClient('mongodb://localhost:27017')
db = client['DataBase_Lanjut']
collection = db['TUGAS1']

# CSV file path
csv_file_path = "database_project\orders.csv"

with open(csv_file_path, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        collection.insert_one(row)