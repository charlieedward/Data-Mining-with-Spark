'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

import sys
import time
import random
from blackbox import BlackBox


input_file_path = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file_path = sys.argv[4]

start_time = time.time()

random.seed(553)

sample = list()
res = list()
bx = BlackBox()
n = 100

for i in range(num_of_asks):
    stream_user = bx.ask(input_file_path, stream_size)
    if i == 0:
        # for the first 100 users, we could directly save them
        for user in stream_user:
            sample.append(user)
    else:
        for user in stream_user:
            n += 1
            prob = random.random()
            # if the probability is less than s/n, we accept the sample
            if prob < 100 / n:
                sample[random.randint(0, 99)] = user
    res.append(str((i+1)*100) + "," + sample[0] + "," + sample[20] + "," + sample[40] + "," + sample[60] + "," + sample[80] + "\n")

with open(output_file_path, 'w') as f:
    f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    for i in range(len(res)):
        f.write(res[i])

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)