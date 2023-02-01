'''
Student: KuoChenHuang
USCID: 8747-1422-96
'''

from pyspark import SparkContext
import sys
import time
import itertools
from collections import defaultdict

threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweeness_file_path = sys.argv[3]
output_file_path = sys.argv[4]

def Girvan_Newman(root, graph):

    # ===================== Build The Tree =====================
    tree = defaultdict(set)
    tree[0].add(root)

    visited = set()  # set for visited nodes.
    visited.add(root)  # visited -> (E)

    neighbors = graph[root]
    children = defaultdict(set)
    parent = defaultdict(set)
    path_to_node = defaultdict(int)

    path_to_node[root] = 1
    for neighbor in neighbors:
        path_to_node[neighbor] = 1
        parent[neighbor].add(root)
        children[root].add(neighbor)

    level = 1
    #print('root:' ,root)
    while neighbors:
        #print(neighbors)
        tree[level] = neighbors

        for neighbor in neighbors:
            visited.add(neighbor)  # visited -> (E,D,F)

        next_level = set() # to store the nodes we iterate next round
        for neighbor in neighbors: # -> D/F
            for n_n in graph[neighbor]: # -> D: B/G/E/F   F:G/D/E
                if n_n not in visited:  # D:B/G   F:G
                    path_to_node[n_n] += path_to_node[neighbor]
                    parent[n_n].add(neighbor)
                    children[neighbor].add(n_n)
                    next_level.add(n_n)
        level += 1
        neighbors = next_level
    #print('tree: ', tree)
    #print('path_to_node: ', path_to_node)
    #print('parent: ', parent)
    #print('children: ', children)

    # ===================== Calculate Betweeness =====================

    # Initialize
    point = defaultdict(float)
    edge_weight = defaultdict(float)
    for node in graph:
        point[node] = 1

    total_level = len(tree)
    #print('total_level: ', total_level)

    # no need to modify the points in the last level, so only need to iterate through level 1~n
    for level in range(1, total_level)[::-1]:
        node_in_level = tree[level]
        for node in node_in_level:
            for parent_node in parent[node]:
                point_to_parent = point[node] * path_to_node[parent_node] / path_to_node[node]
                edge = tuple(sorted((node, parent_node)))
                edge_weight[edge] = point_to_parent
                point[parent_node] += point_to_parent

    #print('point: ', point)
    #print('edge_weight: ', edge_weight)
    return [(edge, weight) for edge, weight in edge_weight.items()]



def find_community(vertex, graph):
    visited = set()
    all_community = list()
    for node in vertex:
        if node in visited:
            continue
        else:
            visited.add(node)

        neighbors = graph[node]
        community = set()
        community.add(node)

        if len(neighbors) == 0:
            all_community.append(community)
            continue

        while neighbors:
            neighnors_of_neighbors_set = set() # to store the node we need to iterate in the next round
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                else:
                    visited.add(neighbor)
                    community.add(neighbor)
                    neighnors_of_neighbors = graph[neighbor]
                    for n_n in neighnors_of_neighbors:
                        if n_n in visited:
                            continue
                        else:
                            neighnors_of_neighbors_set.add(n_n)
            neighbors = neighnors_of_neighbors_set
        all_community.append(community)
    return all_community

def calculate_modularityQ(m, A, k, community):
    Q = 0.0
    for s in community:
        for i in s:
            for j in s:
                Q += A[(i, j)] - (k[i] * k[j]) / (2 * m)
    Q = Q / (2 * m)

    return Q


def get_best_community(vertex, betweenness, edge, graph):
    # betweenness -> the first time betweenness(all the nodes and edges)
    round = 1
    continue_divide = True
    best_Q = -1  # Q range: [-1,1]
    optimal_community = list()

    # get m, k, A from ORIGINAL graph
    # ==================== m ====================
    m = len(edge) / 2  # since graph is directed, we need to divide by 2 to get the actual numbers of edge
    M = m # to store the edges of the graph in each round
    # ==================== A ====================
    A = defaultdict(int)  # to record if edge exist
    for pair in edge:
        # A[] = 1 -> the actual existence of the edge
        A[(pair[0], pair[1])] = 1
        A[(pair[1], pair[0])] = 1
    # ==================== k ====================
    k = defaultdict(int)
    for key, value in graph.items():
        k[key] = len(value)

    # ==================== Create Communities ====================
    while continue_divide:

        if round == 1:
            my_betweenness = betweenness
            round = 2

        else:
            my_betweenness = vertex_rdd.map(lambda vertex: Girvan_Newman(vertex, graph)).\
                            flatMap(lambda res: [pair for pair in res]). \
                            reduceByKey(lambda x, y: x + y). \
                            map(lambda x: (x[0], x[1] / 2)). \
                            sortBy(lambda x: (-x[1], x[0])).collect()
        '''
        my_betweenness = vertex_rdd.map(lambda vertex: Girvan_Newman(vertex, graph)). \
                         flatMap(lambda res: [pair for pair in res]). \
                        reduceByKey(lambda x, y: x + y). \
                        map(lambda x: (x[0], x[1] / 2)). \
                        sortBy(lambda x: (-x[1], x[0])).collect()
        '''
        highest_betweenness = my_betweenness[0][1]

        # find the highest betweenness and vut the edge
        for pair in my_betweenness:
            if pair[1] == highest_betweenness:
                node1 = pair[0][0]
                node2 = pair[0][1]
                graph[node1].remove(node2)
                graph[node2].remove(node1)
                M -= 1
            else:
                break

        community = find_community(vertex, graph)
        Q = calculate_modularityQ(m, A, k, community)
        #print('Q: ', Q)
        if Q > best_Q:
            best_Q = Q
            optimal_community = community
        #print('best_Q:  ', best_Q)

        #print('m: ',m)
        if M == 0:
            continue_divide = False

    return optimal_community


start_time = time.time()

sc = SparkContext('local[*]', 'HW4_task2')
sc.setLogLevel('ERROR')

rdd = sc.textFile(input_file_path)
header = rdd.first()
# rdd -> [['user_id', 'business_id']...]
rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(',')).cache()

# user_list
user_list = rdd.map(lambda x: x[0]).distinct().collect()
# return the key-value pairs -> key: user, value: list of business which user has rated
user_rate = rdd.groupByKey().mapValues(list).collectAsMap()
#print(user_rate['LcCRMIDz1JgshpPGYfLDcA'])

# generate the vertices and edges
vertex = set()
edge = set()
for pair in itertools.combinations(user_list, 2):
    if len(set(user_rate[pair[0]]).intersection(set(user_rate[pair[1]]))) >= threshold:
        # an undirected edge can be replaced by two directed edges
        edge.add((pair[0], pair[1]))
        edge.add((pair[1], pair[0]))
        vertex.add(pair[0])
        vertex.add(pair[1])

vertex_rdd = sc.parallelize(vertex)

graph = defaultdict(set)
# graph e.g. -> graph = {'5' : ('3','7'), '3' : ('2', '4')....}
for pair in edge:
    graph[pair[0]].add(pair[1])

# ======================= Betweenness =======================

betweenness = vertex_rdd.map(lambda vertex: Girvan_Newman(vertex, graph)).\
            flatMap(lambda res: [pair for pair in res]). \
            reduceByKey(lambda x, y: x + y). \
            map(lambda x: (x[0], x[1] / 2)). \
            sortBy(lambda x: (-x[1], x[0])).collect()


#o_betweenness = betweenness.collect()
#print(betweenness)

with open(betweeness_file_path, 'w') as f:
    for pair in betweenness:
        f.write(str(pair[0]) + "," + str(round(pair[1], 5)) + "\n")

# ======================= Community Detection =======================
community = get_best_community(vertex, betweenness, edge, graph)
community = sc.parallelize(community)
res = community.map(lambda x: sorted(x)).sortBy(lambda x: (len(x),x)).collect()
#print(res)

with open(output_file_path, 'w') as f:
    for community in res:
        f.write(str(community)[1:-1] + "\n")

end_time = time.time()
total_time = end_time - start_time
print('Duration: ', total_time)