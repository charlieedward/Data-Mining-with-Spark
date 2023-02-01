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

input_file_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file_path = sys.argv[4]

def generate_hash_parameters(num):
    a_list = random.sample(range(1, sys.maxsize - 1), num)
    b_list = random.sample(range(1, sys.maxsize - 1), num)
    return a_list, b_list

def myhashs(s):
    result = []
    #a_list, b_list = generate_hash_parameters(2)
    user_code = int(binascii.hexlify(s.encode('utf8')), 16)
    for (a, b) in zip(a_list, b_list):
        hash_value = ( (a * user_code + b) % 115249) % 69997
        result.append(hash_value)
    return result

def calculate_fpr(previous_users, temp_users):
    N = 0
    False_num = 0
    for user in temp_users:
        hash_value = myhashs(user)
        # we only need to consider the case that user has not shown yet(but we say it has)
        if user not in previous_users:
            N += 1
            shown_we_say = True
            for hv in hash_value:
                if filter[hv] == 0:
                    shown_we_say = False
            if shown_we_say == True:
                False_num += 1

        for hv in hash_value:
            filter[hv] = 1

    #print('False_num: ', False_num)
    #print('N: ', N)
    return False_num/N


start_time = time.time()

filter = [0] * 69997
bx = BlackBox()
# keep track of previous users
previous_users = set()
res = list()
a_list, b_list = generate_hash_parameters(2)

for i in range(num_of_asks):
    temp_users = bx.ask(input_file_path, stream_size)
    fpr = calculate_fpr(previous_users, temp_users)
    previous_users = previous_users.union(set(temp_users))
    res.append((i, fpr))
#print(res)

with open(output_file_path, 'w') as f:
    f.write("Time,FPR\n")
    for (i, fpr) in res:
        f.write(str(i) + "," + str(fpr))
        f.write("\n")
    f.close()

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)
