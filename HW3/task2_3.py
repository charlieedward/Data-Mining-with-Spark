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
import math
from math import sqrt
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

def calculate_weight(business_1, business_2, business_based_data):
    business_1_data = business_based_data[business_1]
    business_2_data = business_based_data[business_2]

    # co-rater: a list containing the users who both rated business_1 and business_2
    co_rater = list(set(business_1_data.keys()).intersection(set(business_2_data.keys())))

    num_of_co_rater = len(co_rater)
    if num_of_co_rater <= 15:
        return 0.001

    # here i use the 'co-rated' to calculate the avg
    business_1_numerator = 0
    business_2_numerator = 0
    for user in co_rater:
        business_1_numerator += business_1_data[user]
        business_2_numerator += business_2_data[user]

    business_1_avg = business_1_numerator / num_of_co_rater
    business_2_avg = business_2_numerator / num_of_co_rater

    # Pearson Formula
    numerator = 0
    denominator_1 = 0
    denominator_2 = 0

    for user in co_rater:
        numerator += ((business_1_data[user] - business_1_avg) * (business_2_data[user] - business_2_avg))
        denominator_1 += ((business_1_data[user] - business_1_avg) ** 2)
        denominator_2 += ((business_2_data[user] - business_2_avg) ** 2)

    if denominator_1 == 0 or denominator_2 == 0:
        return 0

    else:
        denominator = sqrt(denominator_1) * sqrt(denominator_2)
        weight = numerator / denominator
        #print('weight: ', weight)
        #weight = float(weight * (math.pow(abs(weight), 5)))

        return weight


def item_based_cf(pair, user_avg, business_avg, user_based_data, business_based_data):
    # pair: the {'user_id', 'business_id'} pair which needs a prediction
    # user_avg: {'user_id':avg....} -> {'o0p-iTC5yTBV5Yab_7es4g': 4.0, '=-qj9ouN0bzMXz1vfEslG-A': 2.0,...}
    # business_avg: {'business_id':avg....}
    # user_based_data: {'user_id': {'business_id': stars}} -> e.g {'o0p-iTC5yTBV5Yab_7es4g': {'iAuOpYDfOTuzQ6OPpEiGwA': 4.0}, .....}
    # business_based_data: {business_id: {'user_id': stars}} -> e.g {'BgGzWcWPwJ7pN5Bm78BuuA': {'jHX2qMpRIg-W32vJgi50lw': 4.0}, .....}

    user_id = pair[0]
    business_id = pair[1]

    # if the business_id appears the first time
    # business(X)
    if business_id not in business_avg:
        # case 1: if the user has rated other business before, then we use the avg rating of the user has rated as prediction of the new business rating
        # business(X), user(O)
        if user_id in user_avg:
            #print('yes')
            return (user_avg[user_id], 0)
        # case 2: if it's about new user and new business, then we use the avg of other business ratings as prediction
        # business(X), user(X)
        else:
            #print('new')
            #return sum(business_avg.values())/len(business_avg)
            return (3, 0)

    # case 3: if there's data about the business, but it's the first time the user rates, then we use the avg of the business ratings as prediction
    # business(O), user(X)
    elif user_id not in user_avg:
        return (business_avg[business_id], 0)

    # case 4:
    # business(O), user(O)
    else:
        # get all the business_id that the user has rated
        business_user_has_rated = user_based_data[user_id].keys()
        # get all the users who have rated the business(we need to find its intersection with other business to calculate the weight)
        user_rased_the_business = business_based_data[business_id].keys()

        # weight_list: to store the weight of every pair
        weight_list = list()

        for business in business_user_has_rated:
            if business != business_id:
                weight = calculate_weight(business, business_id, business_based_data)
                # filter out the negative weight
                if weight > 0:
                    weight_list.append((weight, user_based_data[user_id][business]))

        N = min(12, len(weight_list))
        high_weight = len([x for x in weight_list if x[0] >= 0.1])

        # Final Prediction
        weight_list.sort(key = lambda x: x[0], reverse=True)
        #weight_list.sort(key=lambda x: abs(x[0]), reverse=True)

        topN = weight_list[:N]

        numerator = 0
        denominator = 0

        for data in topN:
            # Case Amplification
            weight = round(data[0],3)
            rating = data[1]
            numerator += (rating * weight)
            denominator += abs(weight)
        if denominator == 0 or numerator == 0:
            prediction = sum(user_based_data[user_id].values())/len(user_based_data[user_id])
        else:
            prediction = numerator / denominator

        return ((0.8 * prediction) + (0.2 * sum(business_based_data[business_id].values())/len(business_based_data[business_id].values())), high_weight)

def prediction_combination(cf_prediction, model_prediction):
    res = list()
    #print('cf_prediction: ', cf_prediction)
    for i in range(len(cf_prediction)):
        user_id = cf_prediction[i][0]
        business_id = cf_prediction[i][1]
        cf_pred = round(float(cf_prediction[i][2][0]),3)
        heigh_weight = int(cf_prediction[i][2][1])

        if heigh_weight <= 3:
            prediction = 0.2 * cf_pred + 0.8 * model_prediction[i]
        elif heigh_weight <= 7:
            prediction = 0.4 * cf_pred + 0.6 * model_prediction[i]
        elif heigh_weight <= 10:
            prediction = 0.55 * cf_pred + 0.45 * model_prediction[i]
        else:
            prediction = 0.7 * cf_pred + 0.3 * model_prediction[i]

        temp = (user_id, business_id, prediction)
        res.append(temp)

    return res

start_time = time.time()

sc = SparkContext('local[*]', 'HW3_task2_3')
sc.setLogLevel("ERROR")

# ======================================= Mobel Based =======================================
# business_rdd -> [('business_id', (review_count, stars))...]
business_rdd = sc.textFile(business_file_path)\
        .map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['review_count'], x['stars']))).cache()
# business_dict -> {'business_id': (review_count, stars), ....}
business_dict = business_rdd.collectAsMap()

# user_rdd -> [('user_id', (review_count, useful, avg_stars, fans))]
user_rdd = sc.textFile(user_file_path)\
     .map(lambda x: json.loads(x)).map((lambda x: (x['user_id'], (x['review_count'], x['useful'], x['average_stars'], x['fans'])))).cache()
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
testing_data = testing_data_rdd.filter(lambda x: x != test_headers). \
    map(lambda x: (x.split(',')[0], x.split(',')[1]))

training_y_list = training_data.map(lambda x: x[2]).collect()
training_Y = np.array(training_y_list)

training_df = make_new_df(training_data.collect(), business_rdd, business_dict, user_rdd, user_dict)
training_X = np.array(training_df)
'''
testing_y_list = testing_data.map(lambda x: x[2]).collect()
testing_Y = np.array(testing_y_list)
'''
testing_df = make_new_df(testing_data.collect(), business_rdd, business_dict, user_rdd, user_dict)
testing_X = np.array(testing_df)

#xgbr = xgb.XGBRegressor(verbosity = 0, n_estimators = 38, max_depth = 6)
xgbr = xgb.XGBRegressor(seed=10)

xgbr.fit(training_X, training_Y)
model_prediction = xgbr.predict(testing_X)


# ======================================= CF =======================================
cf_training_data = training_data_rdd.filter(lambda x: x != train_headers).map(lambda x: x.split(','))


## ================= Data needed from Training Set =================
# Store the information in dictionary(using collectAsMap()) -> collectAsMap(): Return the key-value pairs in this RDD to the master as a dictionary.
# {'user_id': {'business_id': stars}} -> e.g {'o0p-iTC5yTBV5Yab_7es4g': {'iAuOpYDfOTuzQ6OPpEiGwA': 4.0}, .....}
user_based_data = training_data.map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
# {business_id: {'user_id': stars}} -> e.g {'BgGzWcWPwJ7pN5Bm78BuuA': {'jHX2qMpRIg-W32vJgi50lw': 4.0}, .....}
business_based_data = training_data.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
#print(business_rate)

# Calculate the avg of each user and item
# {'user_id':avg....} -> {'o0p-iTC5yTBV5Yab_7es4g': 4.0, '=-qj9ouN0bzMXz1vfEslG-A': 2.0,...}
user_avg = training_data.map(lambda x: (x[0], float(x[2]))).groupByKey().map(lambda x: (x[0], (sum(x[1])/len(x[1])))).collectAsMap()
business_avg = training_data.map(lambda x: (x[1], float(x[2]))).groupByKey().map(lambda x: (x[0], (sum(x[1])/len(x[1])))).collectAsMap()


# [....('qUL3CdRRF1vedNvaq06rIA', 'AYL_y8ahquUW0o-cvIyLbg', (3.3999999999999995, 0))]
cf_prediction = testing_data.map(lambda x: (x[0], x[1], item_based_cf(x, user_avg, business_avg, user_based_data, business_based_data))).collect()
#print(cf_prediction)

# ======================================= Combine 2 Prediction =======================================
final_prediction = prediction_combination(cf_prediction, model_prediction)
#print(final_prediction)

'''
predict_rating = [x[2] for x in final_prediction]
rmse = np.sqrt(MSE(testing_Y, predict_rating))
print("RMSE : % f" %(rmse))
'''

with open(output_file_path, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for pair in final_prediction:
        f.write("{},{},{}".format(pair[0], pair[1], pair[2]))
        f.write('\n')
    f.close()

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)