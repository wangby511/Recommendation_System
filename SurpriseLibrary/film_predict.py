import io

from surprise import KNNBaseline
from surprise import Dataset


def read_item_names():
    """
    获取电影名到电影id 和 电影id到电影名的映射
    """

    file_name = '../Data/MovieLens/ml-100k/u.item'
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid

data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)

# [STDOUT IN COMMAND LINES]
# Estimating biases using als...
# Computing the pearson_baseline similarity matrix...
# Done computing similarity matrix.

# UserWarning: train() is deprecated. Use fit() instead warnings.warn('train() is deprecated. Use fit() instead', UserWarning)
# algo.train(trainset)
algo.fit(trainset)

rid_to_name, name_to_rid = read_item_names()

origin_film_name = 'GoldenEye (1995)'
toy_story_raw_id = name_to_rid[origin_film_name]
# print(toy_story_raw_id)

toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
# print(toy_story_inner_id)
# 一个电影在algo内部的id, 并不一定等于文件中的id

# 找到最近的k = 10个邻居
K = 10
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=K)
print(toy_story_neighbors)

# 从近邻的id映射回电影名称
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print('The ' + str(K) + ' nearest neighbors of ' + origin_film_name + ' are:')
for movie in toy_story_neighbors:
    print(movie)