#-*-coding:utf-8-*-

import numpy as np
import _pickle as cPickle
INPUT_FILE_NAME = 'popular_music_suprise_format.txt'

def compute_similarity(users_songs, user1, user2):
    song_list1 = users_songs[user1]
    song_list2 = users_songs[user2]
    combination = 0
    for song in song_list1:
        if song in song_list2:
            combination += 1
    return combination


def nearestNeighbors(user_id, users_songs, n = 10):
    '''
        排序选出相似度最高的N个邻居
    '''
    users_and_sims = []
    top_similarity = 0
    for user, songs in users_songs.items():
        if user == user_id:
            continue
        similarity = compute_similarity(users_songs, user_id, user)
        users_and_sims.append([user, similarity])
        if similarity > top_similarity:
            top_similarity = similarity
    users_and_sims.sort(key = lambda x: x[1],reverse=True)
    # print (users_and_sims[:n])
    # print ([x[0] for x in users_and_sims[:n]])
    # return [x[0] for x in users_and_sims[:n]]
    print ("nearestNeighbors = ",users_and_sims[:n])
    return users_and_sims[:n]


def topKRecommendations(user_id, users_songs, k = 10):
    '''
        根据最近的N个邻居进行推荐
    '''

    totals = {}
    recommend = dict()

    top_n_nearest_neighbors = nearestNeighbors(user_id, users_songs, 10)

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



def user_based(r_user):
    users = []
    users_songs = dict()
    songs = dict()

    origin_file = open(INPUT_FILE_NAME, 'r')
    for line in origin_file:
        contents = line.strip().split(",")
        user = contents[0]
        item = contents[1]

        songs_list = users_songs.get(user,[])
        songs_list.append(item)
        users_songs[user] = songs_list

        users.append(user)
        # users[user] = users.get(user,[]).append(item)
        # songs[item] = songs.get(item,[]).append(user)
    # print (len(users)) # 1076
    # print (len(songs)) # 50539
    r_user = users[0]

    user_based_songs = topKRecommendations(r_user, users_songs, 10)

    id_name_dic = cPickle.load(open('popular_song.pkl', 'rb'), encoding='utf-8')
    # print("加载歌曲id到曲单名的映射字典完成...")

    print ("将以下歌曲推荐给用户ID：", r_user)

    for [song, _] in user_based_songs:
        if song in id_name_dic:
            print (id_name_dic[song])


if __name__ == "__main__":
    user_based(r_user=None)
    # choose one in the function