import pandas as pd
from pymongo import MongoClient

client = MongoClient()
collection = client["douban"]["tvl"]
data = list(collection.find())
data = [{},{},{}]
df = pd.DataFrame(data)
print(df)