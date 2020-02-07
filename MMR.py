# Maximal Marginal Relevance (a.k.a MMR) 算法目的是减少排序结果的冗余，同时保证结果的相关性。
# 最早应用于文本摘要提取和信息检索等领域。在推荐场景下体现在，给用户推荐相关商品的同时，保证推荐结果的多样性，即排序结果存在着相关性与多样性的权衡。
# https://zhuanlan.zhihu.com/p/102285855

# MMR = Argmax(Di属于R/S) 入[Sim(Di,Q) - (1-入)max(Dj属于S)Sim2(Di,Dj)]
# 这个公式的含义是每次从未选取列表中选择一个Di使得Di和Q
# 的相关性与Di与已选择列表集合的相关性的差值最大，这就同时考虑了最终结果集合的相关性和多样性，通过入因子来平衡。

# 入越大，推荐结果越相关; 入越小，推荐结果多样性越高。

# 计算物品间的相似度
# incomplete part 也可以相似矩阵直接取值
def calculateSimilarity(i, j):
    # return similarityMatrix[i][j]
    return 0

# 计算多样性 Diversity值
def calculateDiversity(user):
    score = 0.0
    for item1 in user:
        for item2 in user:
            if item1 == item2:
                continue
            score += calculateSimilarity(item1, item2)
    length = len(user)
    diversity = 1.0 - score / (0.5 * length * (length - 1))
    return diversity

# 计算MMR
def MMR(itemScoreDict, lambdaConstant=0.5, topN=10):
    s, r = [], list(itemScoreDict.keys())
    while len(r) > 0:
        score = 0.0
        selectOne = None
        for i in r:
            if selectOne == None:
                selectOne = i
            firstPart = itemScoreDict[i]
            secondPart = 0.0
            for j in s:
                sim2 = calculateSimilarity(i, j)
                if sim2 > secondPart:
                    secondPart = sim2
            equationScore = lambdaConstant * (firstPart - (1 - lambdaConstant) * secondPart)
            if equationScore > score:
                score = equationScore
                selectOne = i
        r.remove(selectOne)
        s.append(selectOne)
    return (s, s[:topN])[topN > len(s)]