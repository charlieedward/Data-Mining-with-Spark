'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import pyspark
from pyspark import SparkContext
import sys
import time
import itertools
import os
from pyspark.sql import SQLContext
import graphframes

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
output_file_path = sys.argv[3]

start_time = time.time()

sc = SparkContext('local[*]', 'HW4_task1')
sc.setLogLevel('ERROR')
sqlContext = SQLContext(sc)

rdd = sc.textFile(input_file_path)
header = rdd.first()
# rdd -> [['user_id', 'business_id']...]
rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

# user_list
user_list = rdd.map(lambda x: x[0]).distinct().collect()
# return the key-value pairs -> key: user, value: list of business which user has rated
user_rate = rdd.groupByKey().mapValues(list).collectAsMap()
#print(user_rate['LcCRMIDz1JgshpPGYfLDcA'])

# generate the vertices and edges
vertex = set()
edge = set()
for pair in itertools.combinations(user_list, 2):
    if len(set(user_rate[pair[0]]).intersection(set(user_rate[pair[1]]))) >= threshold:
        edge.add((pair[0], pair[1]))
        edge.add((pair[1], pair[0]))
        vertex.add(pair[0])
        vertex.add(pair[1])

# create graph, returns a new DataFrame that with new specified column names
vertices = sqlContext.createDataFrame([(v,) for v in vertex]).toDF("id")
edges = sqlContext.createDataFrame(list(edge)).toDF("src", "dst")
graph = graphframes.GraphFrame(vertices, edges)
#graph.inDegrees.show()
#graph.outDegrees.show()

community = graph.labelPropagation(maxIter=5)
#result.show()
'''
+--------------------+------------+
|                  id|       label|
+--------------------+------------+
|PoADjvCdEl-oHyWET...|601295421445|
|7WMbeIW3DTQE1HTYy...|523986010132|
|3bo1FXvQnxw2hmO0C...|644245094404|
|mD5_v03iY_sFK2B3W...|369367187466|
'''
res = community.rdd.map(lambda x: (x[1], x[0])).groupByKey()
# sort the user_ids in each community in the lexicographical order
res = res.map(lambda c: sorted(list(c[1])))
# sort by the size of communities and the first user_id in the community in lexicographical order
res = res.sortBy(lambda x: (len(x), x)).collect()
#print(res[:150])


with open(output_file_path, 'w') as f:
    for community in res:
        string = ''
        for user in community:
            string += "'" + str(user) + "', "
        string = string.rstrip().rstrip(',')
        f.write(string + "\n")
    f.close()

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)
