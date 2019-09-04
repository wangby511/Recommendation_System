
import _pickle as cPickle
from collaborative_filtering import collaborative_filter_wrapper
INPUT_FILE_NAME = 'popular_music_suprise_format.txt'

if __name__ == "__main__":
    cl_filter = collaborative_filter_wrapper(INPUT_FILE_NAME)
    current_playlist = '75050697'
    # current_playlist = '326644112'
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