'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''
from blackbox import BlackBox
from pyspark import SparkContext
import sys
import time
import random
import binascii
from collections import defaultdict

input_file_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file_path = sys.argv[4]
hash_times = 10

start_time = time.time()

bx = BlackBox()

res = list()

def generate_hash_parameters(num):
    a_list = random.sample(range(1, sys.maxsize - 1), num)
    b_list = random.sample(range(1, sys.maxsize - 1), num)
    return a_list, b_list

def myhashs(s):
    result = []
    #a_list, b_list = generate_hash_parameters(hash_times)
    user_code = int(binascii.hexlify(s.encode('utf8')), 16)
    for (a, b) in zip(a_list, b_list):
        hash_value = ( (a * user_code + b) % 11939) % 739
        result.append(hash_value)
    return result

def Flajolet_Martin(users):
    max_zeros_of_each_hash = [0] * hash_times
    for user in users:
        user_hash_value = myhashs(user)
        for i in range(len(user_hash_value)):
            trailing_zeros = len(bin(user_hash_value[i]).split('1')[-1])
            max_zeros_of_each_hash[i] = max(max_zeros_of_each_hash[i], trailing_zeros)

    sum = 0
    for i in range(len(max_zeros_of_each_hash)):
        sum += (2 ** max_zeros_of_each_hash[i])

    return int(sum/len(max_zeros_of_each_hash))


a_list, b_list = generate_hash_parameters(hash_times) #([list_of a],[list_of_b])
#print(parameters)

res = list()
for i in range(num_of_asks):
    temp_users = bx.ask(input_file_path, stream_size)
    distinct_users_count = Flajolet_Martin(temp_users)
    res.append((i, stream_size, distinct_users_count))
#print(res)

with open(output_file_path, 'w') as f:
    f.write("Time,Ground Truth,Estimation\n")
    for (i, stream_size, distinct_users_count) in res:
        f.write(str(i) + "," + str(stream_size) + "," + str(distinct_users_count))
        f.write("\n")
    f.close()

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)
