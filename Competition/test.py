from pyspark import SparkContext
import sys
import time
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from operator import add
from math import sqrt
from sklearn.metrics import mean_squared_error as MSE

folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

business_file_path = folder_path + "business_test.json"
review_file_path = folder_path + "review_train.json"

sc = SparkContext('local[*]', 'Competition_Test')
sc.setLogLevel("ERROR")


business_rdd = sc.textFile(business_file_path)\
        .map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['attributes']))).cache()

# business_dict -> {'business_id': (review_count, stars), ....}
business_dict = business_rdd.collectAsMap()

for i in range(len(business_dict)):
    if
