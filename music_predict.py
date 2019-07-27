from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io

from surprise import KNNBaseline, Reader
from surprise import Dataset
import _pickle as cPickle

id_name_dic = cPickle.load(open('popular_playlist.pkl','rb'), encoding='utf-8')
print("加载歌单id到歌单名的映射字典")

name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典")


file_path = 'popular_music_suprise_format.txt'

reader = Reader(line_format='user item rating timestamp', sep=',')

music_data = Dataset.load_from_file(file_path, reader=reader)

print("构建数据集...user:歌单 item:歌曲 rating:1.0 timestamp:1300000")
trainset = music_data.build_full_trainset()
#sim_options = {'name': 'pearson_baseline', 'user_based': False}

print(trainset.n_items) # 50539
print(trainset.n_users) # 1076

print("开始训练模型...")
#sim_options = {'user_based': False}
#algo = KNNBaseline(sim_options=sim_options)
algo = KNNBaseline()
algo.fit(trainset)

current_playlist = list(name_id_dic.keys())[0]
print('current_playlist =',current_playlist)

# 取出近邻
playlist_id = name_id_dic[current_playlist]
print('playlist_id =',playlist_id)
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print('playlist_inner_id =',playlist_inner_id)

playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)
print('playlist_neighbors =',playlist_neighbors)

# 把歌曲id转成歌曲名字
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
                       for inner_id in playlist_neighbors)
playlist_neighbors = (id_name_dic[playlist_id]
                       for playlist_id in playlist_neighbors)

print()
print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
for playlist in playlist_neighbors:
    print(playlist)

# /Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7 /Users/wangboyuan/Desktop/G7304/Recommendation_System/music_predict.py
# 加载歌单id到歌单名的映射字典
# 加载歌单名到歌单id的映射字典
# 构建数据集...user:歌单 item:歌曲 rating:1.0 timestamp:1300000
# 50539
# 1076
# 开始训练模型...
# Estimating biases using als...
# Computing the msd similarity matrix...
# Done computing similarity matrix.
# current_playlist = 别走，你没有来错地方
# playlist_id = 75050697
# playlist_inner_id = 745
# playlist_neighbors = [1, 3, 4, 5, 6, 13, 14, 16, 17, 18]
#
# 和歌单 《 别走，你没有来错地方 》 最接近的10个歌单为：
#
# 当过千评论的华语翻唱遇上“原唱”【更新】
# 【华语】暖心物语 纯白思念
# 〖循环〗单曲循环是强迫症吗？
# 周杰伦地表最强演唱会2017520南京站曲目
# 简单的爱总是那么吸引人
# 『华语/回忆』95后陪伴我中学时期的歌曲
# 对不起，你是一个好人
# 所有的大人，曾经都是小孩
# 有没有一首歌让你泪流满面
# 专属你的周杰伦