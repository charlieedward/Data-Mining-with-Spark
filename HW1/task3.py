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

review_file_path = sys.argv[1]
business_file_path = sys.argv[2]
output_file_qa = sys.argv[3]
output_file_qb = sys.argv[4]

sc = SparkContext('local[*]', 'HW1_task3')

# ================= Question A =================
def items_in_partition(iterator):
    #yield sum(1 for _ in iterator)
    return [len(list(iterator))]

review_RDD = sc.textFile(review_file_path).map(lambda line: json.loads(line)).cache()
business_RDD = sc.textFile(business_file_path).map(lambda line: json.loads(line)).cache()

def join_and_average(review_RDD, business_RDD):
    review_info_RDD = review_RDD.map(lambda x: (x['business_id'], x['stars']))
    business_info_RDD = business_RDD.map(lambda x: (x['business_id'], x['city']))

    # Join two json data
    join_RDD = review_info_RDD.join(business_info_RDD) # return format like ('Mem13A3C202RzT53npn4NA', (5.0, 'McKees Rocks'))

    join_RDD = join_RDD.map(lambda x: (x[1][1], x[1][0])) # key:city, value: stars

    # Calculate the average
    #res = join_RDD.groupByKey().mapValues(lambda x: sum(x) / len(x)).collect()
    res = join_RDD.mapValues(lambda x: (x, 1)).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])).mapValues(lambda v: v[0]/v[1])
    return res


res = join_and_average(review_RDD, business_RDD)

# Sorting
res = res.sortBy(lambda x: (-x[1], x[0])).collect()
#print(res)

with open(output_file_qa, 'w') as output_file_a:
    output_file_a.write(("city,stars\n"))
    for cityandstar in res:
        output_file_a.write(cityandstar[0] + ',' + str(cityandstar[1]) + '\n')

# ================= Question B =================
# ===== Python =====
start_time_python = time.time()
data = join_and_average(review_RDD, business_RDD).collect()
data = sorted(data, key=lambda x: (-x[1], x[0]))[:10]
print(data)
end_time_python = time.time()
total_time_python = end_time_python - start_time_python
#print(total_time_python)

# ===== Spark =====
start_time_spark = time.time()
data = join_and_average(review_RDD, business_RDD)
data = data.takeOrdered(10, key = lambda x: [-x[1], x[0]])
print(data)
end_time_spark = time.time()
total_time_spark = end_time_spark - start_time_spark
#print(total_time_spark)

res_b = dict()
res_b["m1"] = total_time_python
res_b["m2"] = total_time_spark
res_b["reason"] = "method 2(spark) is faster than method 1(python), because spark runs the program in parallel."

with open(output_file_qb, 'w') as output_file_b:
    json.dump(res_b, output_file_b)
