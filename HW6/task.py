'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

from pyspark import SparkContext
import sys
import time
import random
from sklearn.cluster import KMeans
from collections import defaultdict
import numpy as np
from itertools import combinations


input_file_path = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file_path = sys.argv[3]

def add_element(x1, x2):
    res = list()
    for (feature_of_x1, feature_of_x2) in zip(x1, x2):
        res.append(feature_of_x1 + feature_of_x2)
    return res

def getClusterNum(cluster):
    sum = 0
    for k, v in cluster.items():
        sum = sum + v['N']
    return sum

def mahalanobis_distance(point, cluster):
    point = np.array(list(point[1][1:]))
    c = cluster["SUM"] / cluster["N"]
    variance = (cluster["SUMSQ"] / cluster["N"]) - (cluster["SUM"] / cluster["N"]) ** 2
    sigma = variance ** (1/2)

    z = (point - c) / (sigma)
    m_distance = np.dot(z, z) ** (1 / 2)
    return m_distance

def mahalanobis_distance_by_clusters(cluster1, cluster2):
    centroid_1 = cluster1["SUM"] / cluster1["N"]
    centroid_2 = cluster2["SUM"] / cluster2["N"]

    variance_1 = (cluster1["SUMSQ"] / cluster1["N"]) - (cluster1["SUM"] / cluster1["N"]) ** 2
    variance_2 = (cluster2["SUMSQ"] / cluster2["N"]) - (cluster2["SUM"] / cluster2["N"]) ** 2

    sigma = (variance_2 ** (1 / 2) + variance_1 ** (1 / 2)) / 2
    z = (centroid_1 - centroid_2) / sigma
    m = np.dot(z, z) ** (1 / 2)
    return m


def update_statistics(set_type, cluster, point, np_point):

    set_type[cluster]["points"].append(point)

    np_point = np_point[1][1:]
    set_type[cluster]["N"] += 1
    set_type[cluster]["SUM"] = np.array(add_element(set_type[cluster]["SUM"], np_point))
    point_power_2 = [fea ** 2 for fea in np_point]
    set_type[cluster]["SUMSQ"] = np.array(add_element(set_type[cluster]["SUMSQ"], point_power_2))

start_time = time.time()

sc = SparkContext('local[*]', 'HW6')
sc.setLogLevel('ERROR')

result = ["The intermediate results:\n"]

# Step 0. Preprocessing the data
data_rdd = sc.textFile(input_file_path).map(lambda x: [int(fea) for fea in x.split(',')[:2]] + [float(fea) for fea in x.split(',')[2:]])
# data -> [(index, rest of features)...]
original_data = data_rdd.map(lambda x: (x[0], tuple(x[1:]))).collect()
random.shuffle(original_data)
# N -> total number of data(322312)
N = len(original_data)

# Step 1. Load 20% of the data randomly.(64462)
sample_size = int(0.2 * N)
sample_data = original_data[:sample_size]
#print('Data_len: ', len(sample_data))

# Step 2. Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
k_means = KMeans(n_clusters = n_cluster * 10).fit([data[1][1:] for data in sample_data])
#print('k_means.labels_: ', k_means.labels_)

# Step 3. In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
target_label = sc.parallelize(k_means.labels_).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)\
                .filter(lambda x: x[1] == 1).map(lambda x: x[0]).collect()
#print('target_label: ', target_label)
RS = set()
for data, label in zip(sample_data, k_means.labels_):
    if label in target_label:
        RS.add(data)
#print('RS_len: ', len(RS))

# Step 4. Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
sample_data_new = [data for data in sample_data if data not in RS]
#print('NewData_len: ', len(sample_data_new))
k_means = KMeans(n_clusters = n_cluster).fit([data[1][1:] for data in sample_data_new])

# Step 5. Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
DS = defaultdict(dict)
cluster_data_pair = tuple(zip(k_means.labels_, sample_data_new))
# data_in_cluster -> {cluster_index(ex. 0): list of data(ex. [(254, (,...)),(...)])}
data_in_cluster = sc.parallelize(cluster_data_pair).groupByKey().mapValues(list).collectAsMap()
# N -> {cluster_index: data counts in the cluster}
DS_N = sc.parallelize(cluster_data_pair).map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
# SUM -> {cluster_index: list of features sum} ex.{0: [-1684.1793, -3476.045457, 2070.565...], 1:[..]}
DS_SUM = sc.parallelize(cluster_data_pair).mapValues(lambda x: x[1][1:]).reduceByKey(lambda a, b: add_element(a,b)).collectAsMap()
DS_SUMSQ = sc.parallelize(cluster_data_pair).mapValues(lambda x: [fea ** 2 for fea in x[1][1:]]) \
    .reduceByKey(lambda a,b: add_element(a, b)).collectAsMap()

for label in k_means.labels_:
    DS[label]["points"] = data_in_cluster[label]
    DS[label]["N"] = np.array(DS_N[label])
    DS[label]["SUM"] = np.array(DS_SUM[label])
    DS[label]["SUMSQ"] = np.array(DS_SUMSQ[label])


# Step 6. Run K-Means on the points in the RS with a large K (e.g., 5 times of the number of the input clusters)
# to generate CS (clusters with more than one points) and RS (clusters with only one point).

# we need to generate new CS and RS in this step(RS = RS_new + CS)
RS_new = set()
CS = set()
CS_statistics = defaultdict(dict)

if RS:
    try:
        k_means = KMeans(n_clusters = n_cluster * 5).fit([data[1][1:] for data in RS])
    except:
        k_means = KMeans(n_clusters = min(len(RS), n_cluster)).fit([row[1][1:] for row in RS])

    RS_target_label = sc.parallelize(k_means.labels_).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)\
                    .filter(lambda x: x[1] == 1).map(lambda x: x[0]).collect()

    # RS_data_pair -> (label, data) for every data in RS
    RS_data_pair = tuple(zip(k_means.labels_, RS))

    for data in RS_data_pair:
        if data[0] in RS_target_label:
            RS_new.add(data[1])
        else:
            CS.add(data)

    data_in_CS_cluster = sc.parallelize(CS).groupByKey().mapValues(list).collectAsMap()
    CS_N = sc.parallelize(CS).map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
    CS_SUM = sc.parallelize(CS).mapValues(lambda x: x[1][1:]).reduceByKey(lambda a, b: add_element(a,b)).collectAsMap()
    CS_SUMSQ = sc.parallelize(CS).mapValues(lambda x: [fea ** 2 for fea in x[1][1:]]) \
        .reduceByKey(lambda a,b: add_element(a, b)).collectAsMap()

    for label in CS_N.keys():
        CS_statistics[label]["points"] = data_in_CS_cluster[label]
        CS_statistics[label]["N"] = np.array(CS_N[label])
        CS_statistics[label]["SUM"] = np.array(CS_SUM[label])
        CS_statistics[label]["SUMSQ"] = np.array(CS_SUMSQ[label])
    #print(CS_statistics)

    result += "Round 1: " + str(getClusterNum(DS)) + "," + str(len(CS_statistics)) + "," + \
               str(getClusterNum(CS_statistics)) + "," + str(len(RS_new)) + '\n'

else:
    result += "Round 1: " + str(getClusterNum(DS)) + "," + str(0) + "," + str(0) + "," + str(0) + '\n'

'''
with open(output_file_path, 'w') as f:
    f.write("The intermediate results:\n")
    # ( the number of the DS points, the number of the clusters in the CS, the number of the CS points, and the number of the points in the RS set.)
    f.write("Round 1: " +  str(getClusterNum(DS)) + "," + str(len(CS_statistics)) + "," + str(getClusterNum(CS_statistics)) + "," + str(len(RS_new)) + '\n')
'''

for round in [2, 3, 4, 5]:

    # Step 7. Load another 20% of the data randomly
    if round == 5:
        collected_data = original_data[int(0.2 * N) * 4:]
    else:
        collected_data = original_data[(round - 1) * int(0.2 * N): round * int(0.2 * N)]

    # Step 8. For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them to the
    # nearest DSclusters if the distance is <2 root 𝑑.
    d = len(collected_data[0][1][1:])
    '''
    DS_index = set() # keep track of points that's already assigned to DS
    CS_index = set() # keep track of points that's already assigned to CS
    RS_index = set() # keep track of points that's already assigned to RS
    '''

    for i in range(len(collected_data)):
        point = collected_data[i]
        np_point = np.array(collected_data[i],dtype=object)
        ds_min_dist = sys.maxsize
        ds_cluster = 0
        for cluster in DS:
            #print('cluster: ',cluster)
            dist = mahalanobis_distance(np_point, DS[cluster])
            #print(dist)
            if dist < ds_min_dist:
                ds_min_dist = dist
                ds_cluster = cluster

        # if belongs to DS

        if ds_min_dist < 2 * d ** (1 / 2):
            #DS_index.add(i)
            update_statistics(DS, ds_cluster, point, np_point)

        #   Step 9. For the new points that are not assigned to DS clusters,
        #   using the Mahalanobis Distance and assign the points to the nearest CS clusters if the distance is < 2 root 𝑑.

        # points that do not belong to DS

        else:
            cs_min_dist = sys.maxsize
            cs_cluster = 0
            for cluster in CS_statistics:
                dist = mahalanobis_distance(np_point, CS_statistics[cluster])
                if dist < cs_min_dist:
                    cs_min_dist = dist
                    cs_cluster = cluster

            if cs_min_dist < 2 * d ** (1 / 2):
                #CS_index.add(i)
                update_statistics(CS_statistics, cs_cluster, point, np_point)

            # Step 10. For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
            else:
                #RS_index.add(i)
                #print('add1')
                RS_new.add(collected_data[i])

    '''
    print('len_DS: ', sum([DS[a]['N'] for a in DS]))
    print('len_CS: ', sum([CS_statistics[a]['N'] for a in CS_statistics]))
    print('len_RS: ', len(RS_new))
    '''

    # Step 11. Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters)
    # to generate CS (clusters with more than one points) and RS (clusters with only one point).
    if RS_new:
        try:
            k_means = KMeans(n_clusters = n_cluster * 5).fit([data[1][1:] for data in RS_new])
        except:
            k_means = KMeans(n_clusters = min(len(RS_new), n_cluster)).fit([data[1][1:] for data in RS_new])

        RS_target_label = sc.parallelize(k_means.labels_).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)\
                        .filter(lambda x: x[1] == 1).map(lambda x: x[0]).collect()

        #print('RS_target_label: ', RS_target_label)
        CS_this_round = set()

        #print('len_RS_new_1: ', len(RS_new))
        RS_nes_list = list(RS_new)
        for data, label in zip(RS_new, k_means.labels_):
            # if still belongs to RS
            if label in RS_target_label:
                continue
            # if belongs to CS now
            else:
                # remove it from RS
                RS_nes_list.remove(data)
                label = label * 553 # to make the label different from the earlier ones
                x = tuple((label, data))
                # add it to new CS set
                CS_this_round.add(x)

        RS_new = set(RS_nes_list)
        #print('len_RS_new_2: ', len(RS_new))

        new_data_in_CS_cluster = sc.parallelize(CS_this_round).groupByKey().mapValues(list).collectAsMap()
        new_CS_N = sc.parallelize(CS_this_round).map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b).collectAsMap()
        new_CS_SUM = sc.parallelize(CS_this_round).mapValues(lambda x: x[1][1:]).reduceByKey(lambda a, b: add_element(a, b)).collectAsMap()
        new_CS_SUMSQ = sc.parallelize(CS_this_round).mapValues(lambda x: [fea ** 2 for fea in x[1][1:]]) \
            .reduceByKey(lambda a, b: add_element(a, b)).collectAsMap()

        for label in new_CS_N.keys():
            CS_statistics[label]["points"] = new_data_in_CS_cluster[label]
            CS_statistics[label]["N"] = np.array(new_CS_N[label])
            CS_statistics[label]["SUM"] = np.array(new_CS_SUM[label])
            CS_statistics[label]["SUMSQ"] = np.array(new_CS_SUMSQ[label])

    # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2 root 𝑑.
    while True:
        compare_list = list(combinations(list(CS_statistics.keys()), 2))
        old_cluster = set(CS_statistics.keys())
        merge_list = []
        for (idx1, idx2) in compare_list:

            m_distance = mahalanobis_distance_by_clusters(CS_statistics[idx1], CS_statistics[idx2])

            if m_distance < 2 * d ** (1 / 2):
                CS_statistics[idx1]["points"] += CS_statistics[idx2]["points"]
                CS_statistics[idx1]["N"] += CS_statistics[idx2]["N"]
                CS_statistics[idx1]["SUM"] += CS_statistics[idx2]["SUM"]
                CS_statistics[idx1]["SUMSQ"] += CS_statistics[idx2]["SUMSQ"]
                del CS_statistics[idx2]
                break
        new_cluster = set(CS_statistics.keys())
        # no clusters to merge(no update)
        if new_cluster == old_cluster:
            break

    # If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a
    # Mahalanobis Distance < < 2 root 𝑑.
    if round == 5:
        for cs_idx, value in list(CS_statistics.items()):
            dis_list = []
            dis_dic = {}
            min_dist = sys.maxsize
            target_ds_index = 0
            for ds_idx in DS.keys():
                distance = mahalanobis_distance_by_clusters(CS_statistics[cs_idx], DS[ds_idx])
                if distance < min_dist:
                    min_dist = distance
                    target_ds_index = ds_idx

            if min_dist < 2 * d ** (1 / 2):
                DS[target_ds_index]["points"] += CS_statistics[cs_idx]["points"]
                DS[target_ds_index]["N"] += CS_statistics[cs_idx]["N"]
                DS[target_ds_index]["SUM"] += CS_statistics[cs_idx]["SUM"]
                DS[target_ds_index]["SUMSQ"] += CS_statistics[cs_idx]["SUMSQ"]
                del CS_statistics[cs_idx]

    result += "Round " + str(round) + ": "  + str(getClusterNum(DS)) + "," + str(len(CS_statistics)) + "," + str(getClusterNum(CS_statistics)) + "," + str(len(RS_new)) + '\n'

result += ("\n" + "The clustering results:\n")

cluster_res = {}

#rs_count = 0
#cs_count = 0
#ds_count = 0
for rs_point in RS_new:
    #rs_count += 1
    cluster_res[rs_point[0]] = -1

for key, val in CS_statistics.items():
    for cs_point in CS_statistics[key]["points"]:
        #cs_count += 1
        cluster_res[cs_point[0]] = -1

for key, val in DS.items():
    for ds_point in DS[key]["points"]:
        #ds_count += 1
        cluster_res[ds_point[0]] = key


#print('rs_count: ',rs_count)
#print('cs_count: ',cs_count)
#print('ds_count: ',ds_count)

clust_res_out = sorted(cluster_res.items(), key = lambda kv: kv[0])

for (key, val) in clust_res_out:
    result += str(key) + "," + str(val) + "\n"

with open(output_file_path, "w") as f:
    f.writelines(result)

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)