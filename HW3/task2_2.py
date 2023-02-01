'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

from pyspark import SparkContext
import sys
import time
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error as MSE

folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

business_file_path = folder_path + "business.json"
review_file_path = folder_path + "review_train.json"
user_file_path = folder_path + "user.json"

train_file_path = folder_path + "yelp_train.csv"

def make_new_df(original_data, business_rdd, business_dict, user_rdd, user_dict):
    user_review_count_avg = user_rdd.map(lambda x: x[1][0]).mean()
    user_useful_avg = user_rdd.map(lambda x: x[1][1]).mean()
    user_stars_avg = user_rdd.map(lambda x: x[1][2]).mean()
    user_fans_avg = user_rdd.map(lambda x: x[1][3]).mean()

    business_review_count_avg = business_rdd.map(lambda x: x[1][0]).mean()
    business_stars_avg = business_rdd.map(lambda x: x[1][1]).mean()

    new_df = dict()
    new_df['user_review_count'] = list()
    new_df['useful'] = list()
    new_df['user_stars'] = list()
    new_df['fans'] = list()
    new_df['business_review_count'] = list()
    new_df['business_stars'] = list()

    for data in original_data:
        # if there's data about the user in the user.json
        if data[0] in user_dict:
            new_df['user_review_count'].append(user_dict[data[0]][0])
            new_df['useful'].append(user_dict[data[0]][1])
            new_df['user_stars'].append(user_dict[data[0]][2])
            new_df['fans'].append(user_dict[data[0]][3])
        # if not, use the mean as his/her data
        else:
            new_df['user_review_count'].append(user_review_count_avg)
            new_df['useful'].append(user_useful_avg)
            new_df['user_stars'].append(user_stars_avg)
            new_df['fans'].append(user_fans_avg)

        # if there's data about the business in the business.json
        if data[1] in business_dict:
            new_df['business_review_count'].append(business_dict[data[1]][0])
            new_df['business_stars'].append(business_dict[data[1]][1])

    res = pd.DataFrame.from_dict(new_df)

    return res

start_time = time.time()

sc = SparkContext('local[*]', 'HW3_task2_2')
sc.setLogLevel("ERROR")

# business_rdd -> [('business_id', (review_count, stars))...]
business_rdd = sc.textFile(business_file_path)\
        .map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).cache()
# business_dict -> {'business_id': (review_count, stars), ....}
business_dict = business_rdd.collectAsMap()

# user_rdd -> [('user_id', (review_count, useful, avg_stars, fans))]
user_rdd = sc.textFile(user_file_path)\
     .map(lambda x: json.loads(x)).map((lambda x: (x['user_id'], (x['review_count'], x['useful'], x['average_stars'], x['fans'])))).cache()
#print(user_rdd[:10])
# user_dict -> {'user_id': (review_count, useful, avg_stars, fans)}
user_dict = user_rdd.collectAsMap()

training_data_rdd = sc.textFile(train_file_path)
# Skip the first row(header)
train_headers = training_data_rdd.first()
# training_data -> [('user_id', 'business_id', 'stars')]
training_data = training_data_rdd.filter(lambda x: x != train_headers). \
    map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))

testing_data_rdd = sc.textFile(test_file_path)
# Skip the first row(header)
test_headers = testing_data_rdd.first()
# testing_data -> [('user_id', 'business_id')]
'''
testing_data = testing_data_rdd.filter(lambda x: x != test_headers). \
    map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))
'''
testing_data = testing_data_rdd.filter(lambda x: x != test_headers). \
    map(lambda x: (x.split(',')[0], x.split(',')[1]))

training_y_list = training_data.map(lambda x: x[2]).collect()
training_Y = np.array(training_y_list)

training_df = make_new_df(training_data.collect(), business_rdd, business_dict, user_rdd, user_dict)
training_X = np.array(training_df)

'''
testing_data_2 = testing_data_rdd.filter(lambda x: x != test_headers). \
    map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))
testing_y_list = testing_data_2.map(lambda x: x[2]).collect()
testing_Y = np.array(testing_y_list)
'''

testing_df = make_new_df(testing_data.collect(), business_rdd, business_dict, user_rdd, user_dict)
testing_X = np.array(testing_df)

#xgbr = xgb.XGBRegressor(verbosity = 0, n_estimators = 38, max_depth = 6)
xgbr = xgb.XGBRegressor(seed=10)

xgbr.fit(training_X, training_Y)
prediction = xgbr.predict(testing_X)

test_data_pair = testing_data.collect()

with open(output_file_path, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(test_data_pair)):
        f.write("{},{},{}".format(test_data_pair[i][0], test_data_pair[i][1], prediction[i]))
        f.write('\n')
    f.close()


rmse = np.sqrt(MSE(testing_Y, prediction))
print("RMSE : % f" %(rmse))


end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)