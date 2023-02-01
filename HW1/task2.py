'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import pyspark
from pyspark import SparkContext
from operator import add
import sys
import json
import time


input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
n_partition = int(sys.argv[3])

sc = SparkContext('local[*]', 'HW1_task2')

# Read JSON file as textfile and then parse each line using map
textRDD = sc.textFile(input_file_path).map(lambda line: json.loads(line)).cache()
defalut_RDD = textRDD.map(lambda x: (x["business_id"], 1))

default = dict()
customized = dict()
res = dict()

def items_in_partition(iterator):
    #yield sum(1 for _ in iterator)
    return [len(list(iterator))]

# =========== Default Method ===========
default["n_partition"] = defalut_RDD.getNumPartitions()
#print(default["n_partition"])

default["n_items"] = defalut_RDD.mapPartitions(items_in_partition).collect()
#print(default["n_items"])

d_start_time = time.time()
top10_business = defalut_RDD.reduceByKey(add).takeOrdered(10, key = lambda x: [-x[1], x[0]])
d_end_time = time.time()

default["exe_time"] = d_end_time - d_start_time
#print(default["exe_time"])

# =========== Customized Method ===========
# Let the RDDs have the same number of partitions, so the join will require no additional shuffling
customized_RDD = textRDD.map(lambda x: (x["business_id"], 1)).partitionBy(n_partition, lambda x: ord(x[0][-1]) % n_partition).cache()

customized["n_partition"] = customized_RDD.getNumPartitions()
#print(customized["n_partition"])

customized["n_items"] = customized_RDD.mapPartitions(items_in_partition).collect()
#print(customized["n_items"])

c_start_time = time.time()
top10_business = customized_RDD.reduceByKey(add).takeOrdered(10, key = lambda x: [-x[1], x[0]])
c_end_time = time.time()

customized["exe_time"] = c_end_time - c_start_time

res["default"] = default
res["customized"] = customized

with open(output_file_path, 'w') as output_file:
    json.dump(res, output_file)