'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import pyspark
from pyspark import SparkContext
import sys
import time
import random
import itertools


input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

min_hash_size = 40
# band * row = min_hash_size
band = 20
row = 2
threshold = 0.5

def f(x):
    return x

def min_hash(user_index_list, a, b, user_count ):
    res = 99999
    for i in range(len(user_index_list)):
        res = min(res, (a * user_index_list[i] + b) % user_count)
    return res

def signature_to_band(signature, band, row):
    result = []
    for i in range(band):
        temp = i * row
        # result -> list of (band_index, hash value of 'piece of signatures')
        # need to hash ot a value so that 'groupby' could work later on
        result.append((i, hash(tuple(signature[temp: (i + 1) * row]))))
    return result

def intersection(lst1, lst2):
    common = [value for value in lst1 if value in lst2]
    return common

def jaccard_similarity(candidate_pair, business_user_list, threshold):
    b1_user = business_user_list[candidate_pair[0]]
    b2_user = business_user_list[candidate_pair[1]]

    common_user_len = len(intersection(b1_user, b2_user))
    total = len(b1_user) + len(b2_user) - common_user_len
    similarity = common_user_len / total


    pair_info = dict()
    # business_id_1 and business_id_2 need to be in alphabetical order
    pair_info['business_id_1'] = min(candidate_pair[0],candidate_pair[1])
    pair_info['business_id_2'] = max(candidate_pair[0],candidate_pair[1])
    pair_info['similarity'] = similarity

    return pair_info

start_time = time.time()

sc = SparkContext('local[*]', 'HW3_task1')
raw_data = sc.textFile(input_file_path)
# Skip the first row(header)
headers = raw_data.first()
raw_data = raw_data.filter(lambda x: x != headers)

# return a list of (business_id, user_id)
review_rdd = raw_data.filter(lambda x: x != headers)\
        .map(lambda x: (x.split(',')[1], x.split(',')[0]))

user_list = review_rdd.map(lambda x: x[1]).sortBy(lambda x: x).distinct().zipWithIndex()
user_dict = user_list.collectAsMap() # -> {.....'zyg4-MFtfPWmwucVazSjfw': 11268, 'zzo--VpSQh8PpsGVeMC1dQ': 11269}
# return a dict, so that when creating a matrix, I could use the key-value pair to get the value(index) by key(user_id)
user_count = user_list.count()

business_list = review_rdd.map(lambda x: x[0]).sortBy(lambda x: x).distinct().zipWithIndex()
business_dict = business_list.collectAsMap() # -> {...... 'zzlZJVkEhOzR2tJOLHcF2A': 24730, 'zzzaIBwimxVej4tY6qFOUQ': 24731}
business_count = business_list.count()

# review_rdd: {business_id, user_id}
# business_user -> build a dict, the key is the business_id and the value is a list of it's related users
business_user = review_rdd.map(lambda x: (x[0], user_dict[x[1]])).groupByKey().map(lambda x: (x[0], list(x[1]))).sortByKey()
business_user_list = business_user.collectAsMap() # -> {... 'zzzaIBwimxVej4tY6qFOUQ': [2494, 10917, 6193, 6431, 10154, 4875]}
#print(business_user_list)

a = random.sample(range(user_count), min_hash_size)
b = random.sample(range(user_count), min_hash_size)

# build a signature_matrix -> (...'--FBCX-N37CMYDfs790Bnw', [163, 519, 1292, 85,...., 135, 529])...]
signature_matrix = business_user.map(lambda x: (x[0], [min_hash(x[1], a[i], b[i], user_count) for i in range(min_hash_size)] ))
#print(signature_matrix.take(5))

# split each signature vector(length of 40) into multiple(10) bands(length of 4)
# row_band_list -> [('--6MefnULPED_I942VcFNA',  [(0, -7965122241587477591), (1, 1730426407130461954)....  (9, -5128408155194319691)]),...]
row_band_list = signature_matrix.mapValues(lambda x: signature_to_band(x, band, row))
#row_band_list = signature_matrix.flatMap(lambda x: [(tuple(chunk), x[0]) for chunk in signature_to_band(x[1], band, row)])

# using flatMapValues to let each tuple in the value(list) become a key-value pair, key is (band_index, piece of signatures) and value is business_id
# ->[((0, -7965122241587477591), '--6MefnULPED_I942VcFNA'), ((1, 1730426407130461954), '--6MefnULPED_I942VcFNA')...]
band_business_pair = row_band_list.flatMapValues(f).map(lambda x: (x[1], x[0]))

# identical_band_list -> a list of lists(each business_id in the list has the same identical band)
# e.g. [['-76didnxGiiMO80BjSpYsQ', 'O1TvPrgkK2bUo5O5aSZ7lw'], ['-A5jntJgFglQ6zwAmOiOMw', 'cTqIuG-fvlQQL0OWzsFdig']....]
identical_band_list = band_business_pair.groupByKey().map(lambda x: list(x[1])).filter(lambda x: len(x) > 1)
#print(identical_band_list.take(5))

# perform combination to generate a list of candidate pair tuples
# e.g. [('-76didnxGiiMO80BjSpYsQ', 'O1TvPrgkK2bUo5O5aSZ7lw'), ('-A5jntJgFglQ6zwAmOiOMw', 'cTqIuG-fvlQQL0OWzsFdig')...]
candidate_pair = identical_band_list.flatMap(lambda list: [pair for pair in itertools.combinations(list, 2)]).distinct()

# res -> [{'business_id_1': '-0dWjxaPKrXAn8urSnkSLA', 'business_id_2': 'z4EIzLJlGd7gyje1Q_hKtw', 'similarity': 0.1}...]
res = candidate_pair.map(lambda x: jaccard_similarity(x, business_user_list, threshold))
res = res.filter(lambda x: x['similarity'] >= 0.5).sortBy(lambda x:(x['business_id_1'], x['business_id_2']))
#print(res.take(5))

#print('-8O4kt8AIRhM3OUxt-pWLg: ', business_user_list['-8O4kt8AIRhM3OUxt-pWLg'])
#print('_p64KqqRmPwGKhZ-xZwhtg: ', business_user_list['_p64KqqRmPwGKhZ-xZwhtg'])

with open(output_file_path, 'w') as f:
    f.write("business_id_1, business_id_2, similarity\n")
    for pair in res.collect():
        f.write(str(pair['business_id_1']) + "," + str(pair['business_id_2']) + "," + str(pair['similarity']) + "\n")
    f.close()


end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)