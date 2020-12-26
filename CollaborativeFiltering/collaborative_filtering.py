#-*-coding:utf-8-*-

import numpy as np

class collaborative_filter_wrapper:
    def __init__(self, INPUT_FILE_NAME):
        self.file_path = INPUT_FILE_NAME
        self.init_load_data()

    def init_load_data(self):
        self.users_list = []
        self.users = dict()
        self.songs = dict()

        origin_file = open(self.file_path, 'r')
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
