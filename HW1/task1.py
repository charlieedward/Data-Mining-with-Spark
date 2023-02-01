'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import pyspark
from pyspark import SparkContext
from operator import add
import sys
import json

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

sc = SparkContext('local[*]', 'HW1_task1')

# Read JSON file as textfile and then parse each line using map
textRDD = sc.textFile(input_file_path).map(lambda line: json.loads(line)).cache()

res = dict()

# Question A: The total number of reviews
res["n_review"] = textRDD.count()
#print("QUESTION_A: ", res["n_review"])

# Question B: The number of reviews in 2018
res["n_review_2018"] = textRDD.filter(lambda x: '2018' in x["date"]).count()
#print("QUESTION_B: ", res["n_review_2018"])

# Question C. The number of distinct users who wrote reviews
res["n_user"] = textRDD.map(lambda x: x["user_id"]).distinct().count()
#print("QUESTION_C: ", res["n_user"])

# Question D. The top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
res["top10_user"] = textRDD.map(lambda x: (x["user_id"], 1)).reduceByKey(add).takeOrdered(10, key = lambda x: [-x[1], x[0]])
#print("QUESTION_D: ", res["top10_user"])

# Question E. The number of distinct businesses that have been reviewed
res["n_business"] = textRDD.map(lambda x: x["business_id"]).distinct().count()
#print("QUESTION_E: ", res["n_business"])

# Question F. The top 10 businesses that had the largest numbers of reviews and the number of reviews they had
res["top10_business"] = textRDD.map(lambda x: (x["business_id"], 1)).reduceByKey(add).takeOrdered(10, key = lambda x: [-x[1], x[0]])
#print("QUESTION_F: ", res["top10_business"])

# Combine answers to a JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(res, output_file)