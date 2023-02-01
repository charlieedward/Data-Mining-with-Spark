'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import pyspark
from pyspark import SparkContext
import sys
import time
import math
import itertools
from operator import add
import collections


def processing_data(raw_data, case):
    if case == '1':  # [user: business]
        data = raw_data.map(lambda x: [x.split(',')[0], x.split(',')[1]]).groupByKey().map(
            lambda x: (x[0], sorted(list(set(x[1]))))) \
            .map(lambda x: x[1]).cache()
        return data

    if case == '2':  # [business: user]
        data = raw_data.map(lambda x: [x.split(',')[1], x.split(',')[0]]).groupByKey().map(
            lambda x: (x[0], sorted(list(set(x[1]))))) \
            .map(lambda x: x[1]).cache()
        return data

def getSingle(baskets):
    res = {}
    for basket in baskets:
        for item in basket:
            item = (item,)
            res[item] = res.get(item, 0) + 1
    return res

def getPair(baskets, candidate_for_now, item_len):
     res = {}
     for basket in baskets:
          single_element = sorted(list(set(basket).intersection(candidate_for_now)))
          all_combination = itertools.combinations(single_element, item_len)
          for item in all_combination:
               res[item] = res.get(item, 0) + 1
     return res

def check_pass_threshold(chunk, threshold):
     res = list()
     for item, count in chunk.items():
          if count >= threshold:
               res.append(item)
     return res

def apriori(iterator, data_count, support):
    frequent_list = list()

    item_len = 1
    check_continue = True
    chunk = list(iterator)
    threshold = math.ceil((len(chunk) / data_count) * support)

    # item_len now = 1
    single_candidate = getSingle(chunk)
    single_candidate = check_pass_threshold(single_candidate, threshold)

    frequent_list.append(single_candidate)

    candidate_for_now = set(item[0] for item in single_candidate)

    # item_len now >= 1
    while check_continue:
        item_len += 1
        all_combination = getPair(chunk, candidate_for_now, item_len)
        frequent_pair = check_pass_threshold(all_combination, threshold)
        if len(frequent_pair) >= 1:
             frequent_list.append(frequent_pair)

        elif len(frequent_pair) == 0:
            check_continue = False

    return frequent_list


def final_count(iterator, candidate):
    chunk = list(iterator)
    count_table = dict()
    for basket in chunk:
        for item in candidate:
            if set(item).issubset(basket):
                count_table[item] = count_table.get(item, 0) + 1

    res = list()
    for item, count in count_table.items():
        res.append((item, count))

    return res


def output_format(rdd):
    output = ''
    len_now = 1
    for item in rdd:
        if len(item) != len_now:
            len_now += 1
            output = output[:-1]  # remove the last comma
            output += "\n\n"

        if len(item) == 1:
            output += str(item).replace(',', '')
            output += ','
        else:
            output += str(item)
            output += ','

    return output[:-1]  # remove the last comma


case_number = sys.argv[1]
support = sys.argv[2]
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

start_time = time.time()

sc = SparkContext('local[*]', 'HW2_task1')
raw_data = sc.textFile(input_file_path)  # textFile() method read an entire CSV record as a String and returns RDD[String]
# print(data_each_line.collect())

# Skip the first row(header)
headers = raw_data.first()
raw_data = raw_data.filter(lambda x: x != headers)
# print(raw_data.collect())

data = processing_data(raw_data, case_number)
data_count = data.count()
# print(data.collect())

candidate = data.mapPartitions(lambda partition: apriori(partition, data_count, float(support))).flatMap(lambda x: x)
candidate = candidate.distinct().sortBy(lambda pair: (len(pair), pair)).collect()
# print(candidate)

frequent_item = data.mapPartitions(lambda partition: final_count(partition, candidate)).reduceByKey(add)
frequent_item = frequent_item.filter(lambda item: item[1] >= float(support)).map(lambda item: item[0]).sortBy(
    lambda pair: (len(pair), pair)).collect()
# print(frequent_item)

output_candidate = output_format(candidate)
output_frequent_item = output_format(frequent_item)
# print(output_candidate)

with open(output_file_path, 'w') as output_file:
    output_file.write(
        'Candidates:' + '\n' + output_candidate + '\n\n' + 'Frequent Itemsets:' + '\n' + output_frequent_item)

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)





