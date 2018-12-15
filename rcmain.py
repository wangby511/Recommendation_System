import gzip
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
import time
tic = time.time()
def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

# for elem in readGz("train.json.gz"):
#     print (elem)
#     break

all_set = []
user_set = set()
item_set = set()
user = defaultdict(list)
item = defaultdict(list)

def CalculatePetersonSimilarity(user1, user2, user):
    res = 0
    r_u1_avg = sum([item[1] for item in user[user1]])/len(user[user1])
    r_u2_avg = sum([item[1] for item in user[user2]])/len(user[user2])
    item_intersec = []
    for item_1 in user[user1]:
        for item_2 in user[user2]:
            if item_1[0] == item_2[0]:
                item_intersec.append(item_1)
                break
    if len(item_intersec) == 0:
        return res
    else:
        for elem in item_intersec:
            res = res + (elem[1] - r_u1_avg) * (elem[1] - r_u2_avg)
    deno1 = 0
    for elem in item_intersec:
        deno1 = deno1 + (elem[1] - r_u1_avg)**2
    deno2 = 0
    for elem in item_intersec:
        deno2 = deno2 + (elem[1] - r_u2_avg)**2
    if deno1 == 0 or deno2 == 0:
        return 0
    return res*1.0/((deno1*deno2)**(1/2))

def CalculateJacardSimilarity_user(user1, user2):
    if user1 not in user_set or user2 not in user_set:
        return 0
    item1 = user[user1]
    item2 = user[user2]
    similarity = 0
    for each_item_1 in item1:
        for each_item_2 in item2:
            if each_item_2[0] == each_item_1[0]:
                similarity = similarity + 1
                break
    item_set = set()
    for item in item1:
        item_set.add(item[0])
    for item in item2:
        item_set.add(item[0])
    return similarity*1.0/len(item_set)

def CalculateJacardSimilarity_item(item1, item2):
    if item1 not in item_set or item2 not in item_set:
        return 1
    user1 = item[item1]
    user2 = item[item2]
    similarity = 0
    for each_user_1 in user1:
        for each_user_2 in user2:
            if each_user_2[0] == each_user_1[0]:
                similarity = similarity + 1
                break
    user_set = set()
    for user_i in user1:
        user_set.add(user_i[0])
    for user_i in user2:
        user_set.add(user_i[0])
    return similarity*1.0/len(user_set)

def BuilderSimilarityUser(user_give, user):
    res = []
    for user_each in user_set:
        if user_each == user_give:
            continue
        if CalculateJacardSimilarity_user(user_give, user_each, user) > 0:
            # print(CalculateJacardSimilarity_user(user_give, user_each, user))
            res.append(user_each)
    return res

i = 0
user_sim = defaultdict(list)
for any_user in user_set:
    i = i + 1
    user_sim[any_user] = BuilderSimilarityUser(any_user, user)
    print("processing ... i =", i, ",any_user =", any_user,":",user_sim[any_user])

def Judge(u, i, user_sim, item):
    for each_user in user_sim[u]:
        for each_item in item[each_user]:
            if each_item == i:
                return True
    return False

def PredictBuy_v3(user_g, item_g, item, user, item_set, user_set):
    if user_g not in user_set or item_g not in item_set:
        return False
    else:
        for each_user in item[item_g]:
            if CalculateJacardSimilarity_user(user_g, each_user[0], user) > 0.2:
                return True
    return False

def main():
    i = 0
    for elem in readGz("train.json.gz"):
        all_set.append(elem)
        user_set.add(elem['reviewerID'])
        item_set.add(elem['itemID'])
        i = i + 1

    # use cosine similarity measure similarity between users and items
    for elem in all_set:
        item_R = []
        user_R = []
        user_R.append(elem['reviewerID'])
        item_R.append(elem['itemID'])
        user_R.append(elem['rating'])
        item_R.append(elem['rating'])
        user[elem['reviewerID']].append(item_R)
        item[elem['itemID']].append(user_R)

    print(f"The total number of users is {len(user)}")
    print(f"The total number of items is {len(item)}")
    f = open("pairs_Purchase.txt")
    f_result = open("output.txt", "w")
    index = 0
    countN1 = 0
    countN2 = 0
    while 1:
        line = f.readline().strip()
        if not line:
            break
        if index == 0:
            f_result.write(line + '\n')
            index = index + 1
            continue
        u = line.split("-")[0]
        i = line.split("-")[1]
        countN1 = countN1 + 1

        a = []
        for elem in user[u]:
            a.append(CalculateJacardSimilarity_item(elem[0],i))

        b = []
        for elem in item[i]:
            b.append(CalculateJacardSimilarity_user(elem[0], u))

        sum_user_sim = np.sum(np.array(a))
        sum_item_sim = np.sum(np.array(b))
        if(sum_user_sim == 0 and sum_item_sim == 0):
            continue
        countN2 = countN2 + 1
        print("a = ", a)
        print("b = ", b)
        print("user_sim[", u, "] = ", user_sim[u], ",user[u] =", user[u], "i = ", i)

        # if Judge(u, i, user_sim, item):
        #     f_result.write(u + '-' + i + ',' + '1'+'\n')
        # else:
        #     f_result.write(u + '-' + i + ',' + '0'+'\n')
    toc = time.time()
    print("countN1 = ",countN1,",countN2 = ",countN2)
    print("Total time is", toc - tic, "s")
    f_result.close()
    f.close()

if __name__ == '__main__':
    main()