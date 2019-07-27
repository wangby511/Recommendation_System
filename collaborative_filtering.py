#-*-coding:utf-8-*-

import numpy as np
import _pickle as cPickle
INPUT_FILE_NAME = 'popular_music_suprise_format.txt'


class collaborative_class_wrapper:
    def __init__(self, INPUT_FILE_NAME):
        self.file_path = INPUT_FILE_NAME
        self.load_data()

    def load_data(self):
        self.users_list = []
        self.users = dict()
        self.songs = dict()

        origin_file = open(INPUT_FILE_NAME, 'r')
        for line in origin_file:
            contents = line.strip().split(",")
            user = contents[0]
            item = contents[1]

            songs_list = self.users.get(user, [])
            songs_list.append(item)
            self.users[user] = songs_list

            if user not in self.users_list:
                self.users_list.append(user)

    def compute_similarity_by_common(self,user1,user2):
        list1 = self.users[user1]
        list2 = self.users[user2]
        combination = 0
        for song in list1:
            if song in list2:
                combination += 1
        return combination


    def nearestNeighbors(self, user_id, n = 10):
        '''
            排序选出相似度最高的N个邻居
        '''
        users_and_sims = []
        for user, items in self.users.items():
            if user == user_id:
                continue
            similarity = self.compute_similarity_by_common(user_id, user)
            users_and_sims.append([user, similarity])
        users_and_sims.sort(key = lambda x: x[1],reverse=True)
        # print (users_and_sims[:n])
        # print ([x[0] for x in users_and_sims[:n]])
        return [x[0] for x in users_and_sims[:n]]
        # nearestNeighbors = []
        # for key in users_and_sims:
        #     nearestNeighbors.append(key[0])
        #
        # print("For user,", user_id, "its nearestNeighbors = ", nearestNeighbors)
        # return nearestNeighbors


    def topKRecommendations(self,user_id, users_songs, k = 10):
        '''
            根据最近的N个邻居进行推荐
        '''

        totals = {}
        recommend = dict()

        top_n_nearest_neighbors = self.nearestNeighbors(user_id, n=10)

        for [neighbor_user, similarity] in top_n_nearest_neighbors:
            for song in users_songs[neighbor_user]:
                recommend[song] = recommend.get(song,0) + similarity

        # sorted(recommend.items(), key=lambda x: x[1], reverse=True)
        top_k_recommend_list = []
        for k,v in recommend.items():
            top_k_recommend_list.append([k,v])
        top_k_recommend_list.sort(key=lambda x: x[1], reverse=True)
        top_k_recommend_list = top_k_recommend_list[:10]

        print("top_k_recommend_list =",top_k_recommend_list)

        return top_k_recommend_list

if __name__ == "__main__":
    cl_filter = collaborative_class_wrapper(INPUT_FILE_NAME)
    current_playlist = '75050697'
    current_playlist = '326644112'
    playlist_neighbors = cl_filter.nearestNeighbors(current_playlist)

    id_name_dic = cPickle.load(open('popular_playlist.pkl', 'rb'), encoding='utf-8')
    print("加载歌单id到歌单名的映射字典")

    name_id_dic = {}
    for playlist_id in id_name_dic:
        name_id_dic[id_name_dic[playlist_id]] = playlist_id
    print("加载歌单名到歌单id的映射字典")

    playlist_neighbors = (id_name_dic[playlist_id]
                          for playlist_id in playlist_neighbors)

    print()
    print("和歌单 《", id_name_dic[current_playlist], "》 最接近的10个歌单为：\n")
    for playlist in playlist_neighbors:
        print(playlist)
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

# 和歌单 《 不悔梦归处，只怨太匆匆。 》 最接近的10个歌单为：
#
# 当过千评论的华语翻唱遇上“原唱”【更新】
# 【华语】暖心物语 纯白思念
# 周杰伦地表最强演唱会2017520南京站曲目
# 『华语/回忆』95后陪伴我中学时期的歌曲
# 暗暗作祟| 不甘朋友不敢恋人
# 专属你的周杰伦
# 你知道思念一个人的滋味吗
# 「华语歌曲」
# 愿形容我的词 别太荒唐
# 致我们终将逝去的青春（华语精选）
