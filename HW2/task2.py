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

def data_preprocessing(input_file_path, new_file_name):
    with open(input_file_path, 'r') as file:
        new_data = []
        for line in file.readlines()[1:]: # remove header
            words = line.split(',')
            date = words[0][:-4] + words[0][-2:]
            date = date.replace('\"', '')
            customer_id = words[1].replace('\"', '').lstrip("0")
            date_customer_id = date + '-' + customer_id
            product_id = words[5].replace('\"', '').lstrip("0")
            new_data.append((date_customer_id, product_id))
        # print(new_data)

    with open(new_file_name, 'w') as new_file:
        new_file.write('DATE-CUSTOMER_ID, PRODUCT_ID')
        new_file.write('\n')
        for line in new_data:
            new_file.write(','.join(line))
            new_file.write('\n')


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
            output = output[:-1] # remove the last comma
            output += "\n\n"

        if len(item) == 1:
            output += str(item).replace(',', '')
            output += ','
        else:
            output += str(item)
            output += ','

    return output[:-1] # remove the last comma

threshold = sys.argv[1]
support = sys.argv[2]
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

#mydata_file = 'customer_product.csv'
start_time = time.time()
sc = SparkContext('local[*]', 'HW2_task2')

data_preprocessing(input_file_path, 'customer_product.csv')

data = sc.textFile('customer_product.csv')
# Skip the first row(header)
headers = data.first()
raw_data = data.filter(lambda x: x != headers)

data = data.map(lambda x: [x.split(',')[0], x.split(',')[1]]).groupByKey().map(lambda x: (x[0], list(x[1])))\
     .filter(lambda x: len(x[1]) > float(threshold)).map(lambda x: (x[0], sorted(list(set(x[1]))))).map(lambda x: x[1]).cache()
data_count = data.count()


candidate = data.mapPartitions(lambda partition: apriori(partition, data_count, float(support))).flatMap(lambda x: x)\
      .distinct().sortBy(lambda pair:(len(pair), pair)).collect()
#print('candidate: ', candidate)

frequent_item = data.mapPartitions(lambda partition: final_count(partition, candidate)).reduceByKey(add)
frequent_item = frequent_item.filter(lambda item: item[1] >= float(support)).map(lambda item: item[0]).sortBy(lambda pair:(len(pair), pair)).collect()
#print('frequent_item: ', frequent_item)

output_candidate = output_format(candidate)
output_frequent_item = output_format(frequent_item)
#print(output_candidate)

with open(output_file_path, 'w') as output_file:
    output_file.write('Candidates:' + '\n' + output_candidate + '\n\n' + 'Frequent Itemsets:' + '\n' + output_frequent_item)


end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)