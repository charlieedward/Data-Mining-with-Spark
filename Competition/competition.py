'''
Student: KuoChenHuang
USCID: 8747-1422-96

Method Description:
In hw3, I found that model-based CF performs better than item-based model a lot,
so in this project I would "use model-based CF only".
About feature selection, I mostly focus on business.json file, and what surprised me the most is LOCTATION(longitude, langtitude),
after including location as my training features, my RSME drops significantly.
Another key point is about the regressor parameters, I had tried many combinations and this one is the best I got.

Error Distribution:
>=0 and <1: 102345
>=1 and <2: 32762
>=2 and <3: 6148
>=3 and <4: 789
>=4: 0

RMSE: 0.977285

Execution Time: 176.65113496780396

'''


from pyspark import SparkContext
import sys
import time
import json
import xgboost as xgb
import pandas as pd
import numpy as np
from operator import add
from sklearn.metrics import mean_squared_error as MSE

folder_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

business_file_path = folder_path + "business.json"
review_file_path = folder_path + "review_train.json"
user_file_path = folder_path + "user.json"
photo_file_path = folder_path + "photo.json"
tip_file_path = folder_path + "tip.json"
checkin_file_path = folder_path + "checkin.json"

train_file_path = folder_path + "yelp_train.csv"

def make_new_df(original_data, default_business, business_dict, user_rdd, user_dict, photo_dict, tip_dict, checkin_dict):
    user_review_count_avg = user_rdd.map(lambda x: x[1][0]).mean()
    user_useful_avg = user_rdd.map(lambda x: x[1][1]).mean()
    user_stars_avg = user_rdd.map(lambda x: x[1][2]).mean()
    user_fans_avg = user_rdd.map(lambda x: x[1][3]).mean()

    new_df = dict()
    new_df['user_review_count'] = list()
    new_df['useful'] = list()
    new_df['user_stars'] = list()
    #new_df['fans'] = list()
    new_df['business_review_count'] = list()
    new_df['business_stars'] = list()
    new_df['business_longitude'] = list()
    new_df['business_latitude'] = list()
    new_df['WiFi'] = list()
    new_df['GoodForKids'] = list()
    new_df['BusinessAcceptsCreditCards'] = list()
    new_df['NoiseLevel'] = list()
    new_df['RestaurantsPriceRange2'] = list()
    new_df['photo'] = list()
    new_df['tip_likes'] = list()
    new_df['checkin_counts'] = list()

    for data in original_data:
        # if there's data about the user in the user.json
        if data[0] in user_dict:
            new_df['user_review_count'].append(user_dict[data[0]][0])
            new_df['useful'].append(user_dict[data[0]][1])
            new_df['user_stars'].append(user_dict[data[0]][2])
            #new_df['fans'].append(user_dict[data[0]][3])
        # if not, use the mean as his/her data
        else:
            new_df['user_review_count'].append(user_review_count_avg)
            new_df['useful'].append(user_useful_avg)
            new_df['user_stars'].append(user_stars_avg)
            #new_df['fans'].append(user_fans_avg)

        # if there's data about the business in the business.json
        if data[1] in business_dict:
            business_data = business_dict[data[1]]
            new_df['business_review_count'].append(business_data[0])
            new_df['business_stars'].append(business_data[1])
            new_df['WiFi'].append(business_data[2][0])
            new_df['GoodForKids'].append(business_data[2][1])
            new_df['BusinessAcceptsCreditCards'].append(business_data[2][2])
            new_df['NoiseLevel'].append(business_data[2][3])
            new_df['RestaurantsPriceRange2'].append(business_data[2][4])
            new_df['business_longitude'].append(business_data[3])
            new_df['business_latitude'].append(business_data[4])
        else:
            new_df['business_review_count'].append(default_business['review_count'])
            new_df['business_stars'].append(default_business['stars'])
            new_df['WiFi'].append(default_business['WiFi'])
            new_df['GoodForKids'].append(default_business['GoodForKids'])
            new_df['BusinessAcceptsCreditCards'].append(default_business['BusinessAcceptsCreditCards'])
            new_df['NoiseLevel'].append(default_business['NoiseLevel'])
            new_df['RestaurantsPriceRange2'].append(default_business['Alcohol'])
            new_df['business_longitude'].append(0.5)
            new_df['business_latitude'].append(0.5)


        if data[1] in photo_dict:
            new_df['photo'].append(photo_dict[data[1]])
        else:
            new_df['photo'].append(np.nan)

        if (data[0], data[1]) in tip_dict:
            key = (data[0], data[1])
            new_df['tip_likes'].append(tip_dict[key])
        else:
            new_df['tip_likes'].append(np.nan)

        if data[1] in checkin_dict:
            new_df['checkin_counts'].append(checkin_dict[data[1]])
        else:
            new_df['checkin_counts'].append(np.nan)

    res = pd.DataFrame.from_dict(new_df)

    return res

def preprocessing_attributes(attributes):
    # there are 188,593 business_id in business.json
    if attributes:
        res = list()
        # ============================ (1)Wifi ============================
        # 49.029 data
        if "WiFi" in attributes.keys():
            if attributes["WiFi"] == 'free':
                res.append(2)
            elif attributes["WiFi"] == 'paid':
                res.append(1)
            elif attributes["WiFi"] == 'no':
                res.append(0)
        else:
            res.append(np.nan)

        # ============================ (2)GoodforKids ============================
        # 64,931 data
        if "GoodForKids" in attributes.keys():
            if attributes["GoodForKids"] == 'True':
                res.append(1)
            elif attributes["GoodForKids"] == 'False':
                res.append(0)
        else:
            res.append(np.nan)

        # ============================ (3)BusinessAcceptsCreditCards ============================
        # 140,391 data
        if "BusinessAcceptsCreditCards" in attributes.keys():
            if attributes["BusinessAcceptsCreditCards"] == 'True':
                res.append(1)
            elif attributes["BusinessAcceptsCreditCards"] == 'False':
                res.append(0)
        else:
            res.append(np.nan)

        # ============================ (4)NoiseLevel ============================
        # 140,391 data
        if "NoiseLevel" in attributes.keys():
            if attributes["NoiseLevel"] == 'quiet':
                res.append(3)
            elif attributes["NoiseLevel"] == 'average':
                res.append(2)
            elif attributes["NoiseLevel"] == 'loud':
                res.append(1)
            elif attributes["NoiseLevel"] == 'very_loud':
                res.append(0)
        else:
            res.append(np.nan)

        # ============================ (5)RestaurantsPriceRange2 ============================
        if "RestaurantsPriceRange2" in attributes.keys():
            value = int(attributes["RestaurantsPriceRange2"])
            res.append(value)
        else:
            res.append(np.nan)

    else:
        res = [np.nan, np.nan, np.nan, np.nan, np.nan]

    return res


sc = SparkContext('local[*]', 'Competition_Test')
sc.setLogLevel("ERROR")
start_time = time.time()

business_rdd = sc.textFile(business_file_path)\
        .map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], (x['review_count'], x['stars'],
                                                    preprocessing_attributes(x['attributes']),
                                                    (float(x["longitude"]))/180 if x["longitude"] is not None else 0,
                                                    (float(x["latitude"]))/90 if x["latitude"] is not None else 0))).cache()

# business_dict -> {'business_id': (review_count, stars, ....}
business_dict = business_rdd.collectAsMap()


default_business = {'review_count': business_rdd.map(lambda x: x[1][0]).mean(),
                    'stars': business_rdd.map(lambda x: x[1][1]).mean(),
                    'WiFi': business_rdd.map(lambda x: x[1][2][0]).filter(lambda x: x>=0).mean(),
                    "GoodForKids": business_rdd.map(lambda x: x[1][2][1]).filter(lambda x: x>=0).mean(),
                    "BusinessAcceptsCreditCards": business_rdd.map(lambda x: x[1][2][2]).filter(lambda x: x>=0).mean(),
                    "NoiseLevel": business_rdd.map(lambda x: x[1][2][3]).filter(lambda x: x>=0).mean(),
                    "RestaurantsPriceRange2": business_rdd.map(lambda x: x[1][2][4]).filter(lambda x: x>=0).mean()}
#print(default_business)

# user_rdd -> [('user_id', (review_count, useful, avg_stars, fans))]

user_rdd = sc.textFile(user_file_path)\
     .map(lambda x: json.loads(x)).map((lambda x: (x['user_id'], (x['review_count'], x['useful'], x['average_stars'], x['fans'])))).cache()
# user_dict -> {'user_id': (review_count, useful, avg_stars, fans)}
user_dict = user_rdd.collectAsMap()


photo_rdd = sc.textFile(photo_file_path).map(lambda r: json.loads(r)).map(lambda x:(x['business_id'],1))
# user_dict -> {'business_id': count, ....}
photo_dict = photo_rdd.reduceByKey(add).collectAsMap()

tip_rdd = sc.textFile(tip_file_path).map(lambda r: json.loads(r)).map(lambda x:((x['user_id'],x['business_id']), x["likes"]))
# user_dict -> {('user_id', 'business_id'): count, ....}
tip_dict = tip_rdd.reduceByKey(add).collectAsMap()

checkin_rdd = sc.textFile(checkin_file_path).map(lambda r: json.loads(r)).map(lambda x:(x['business_id'], len(x["time"]) ))
checkin_dict = checkin_rdd.reduceByKey(add).collectAsMap()

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
    map(lambda x: (x.split(',')[0], x.split(',')[1], float(x.split(',')[2])))

training_y_list = training_data.map(lambda x: x[2]).collect()
training_Y = np.array(training_y_list)


training_df = make_new_df(training_data.collect(), default_business, business_dict, user_rdd, user_dict, photo_dict, tip_dict, checkin_dict)
training_X = np.array(training_df)

testing_y_list = testing_data.map(lambda x: x[2]).collect()
testing_Y = np.array(testing_y_list)

testing_df = make_new_df(testing_data.collect(), default_business, business_dict, user_rdd, user_dict, photo_dict, tip_dict, checkin_dict)
testing_X = np.array(testing_df)

#xgbr = xgb.XGBRegressor(verbosity = 0, n_estimators = 38, max_depth = 6)
#xgbr = xgb.XGBRegressor(seed=10)
#xgbr =  xgb.XGBRegressor(learning_rate = 0.1, max_depth = 5, n_estimators = 700, reg_lambda = 1.5, n_jobs = -1)

xgbr =  xgb.XGBRegressor(learning_rate = 0.1, max_depth = 5, n_estimators = 700, reg_lambda = 1.5, n_jobs = -1)
xgbr.fit(training_X, training_Y)
model_prediction = xgbr.predict(testing_X)


testing_data_pair = testing_data.map(lambda x: (x[0], x[1])).collect()
#predict_rating = [x[2] for x in final_prediction]

rmse = np.sqrt(MSE(testing_Y, model_prediction))
print("RMSE : % f" %(rmse))


with open(output_file_path, 'w') as f:
    f.write("user_id, business_id, prediction\n")
    for i in range(len(testing_data_pair)):
        f.write("{},{},{}".format(testing_data_pair[i][0], testing_data_pair[i][1], model_prediction[i]))
        f.write('\n')
    f.close()
'''
x1 = 0
x2 = 0
x3 = 0
x4 = 0
x5 = 0
for i in range(len(model_prediction)):
    diff = abs(model_prediction[i] - testing_Y[i])
    if  diff >= 0 and diff < 1:
        x1 += 1
    elif diff >= 1 and diff < 2:
        x2 += 1
    elif diff >= 2 and diff < 3:
        x3 += 1
    elif diff >= 3 and diff < 4:
        x4 += 1
    else:
        x5 += 1

print(x1,x2,x3,x4,x5)
'''

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)
