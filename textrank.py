# This is my implementation of the TextRank algorithm to generate summaries from documents.
import math
import operator
import random
import nltk

from binaryheap import MinHeap

with open('data.txt', 'r') as myfile:
    data = myfile.read().replace('\n', '')

sentences = nltk.tokenize.sent_tokenize(data)
print(sentences)


def similarity(s1, s2):
    count = 0
    tokensS1 = nltk.word_tokenize(s1)
    tokensS2 = nltk.word_tokenize(s2)
    for w1 in tokensS1:
        for w2 in tokensS2:
            if w1 == w2:
                count += 1
    simScore = count / (math.log(len(tokensS1)) + math.log(len(tokensS2)))
    return simScore


# This code creates a similarity score mapping between each sentence and the other
simMap = {}

for s1 in sentences:
    simMap[s1] = {}

for s1 in sentences:
    for s2 in sentences:
        if s1 != s2:
            simScore = similarity(s1, s2)
            simMap[s1][s2] = simScore
print(simMap)
print(len(simMap))

# This code uses the similarity score mappings to iteratively calculate the weighted score of each sentence

newScoreMap = {}
oldScoreMap = {}

for s1 in sentences:
    newScoreMap[s1] = random.uniform(0, 10)
for s1 in sentences:
    oldScoreMap[s1] = 0


def iterateGraph():
    print("iterating")
    for s1 in sentences:
        newScore = 0
        outSum = 0
        inSum = 0;
        for s2 in simMap[s1]:
            outSum = outSum + simMap[s1][s2]
        for s2 in simMap[s1]:
            inSum = inSum + (simMap[s2][s1] / outSum) * (newScoreMap[s2])
        newScore = (1 - 0.85) + 0.85 * (inSum)
        oldScoreMap[s1] = newScoreMap[s1]
        newScoreMap[s1] = newScore


def graphTest():
    for s1 in sentences:
        if abs(oldScoreMap[s1] - newScoreMap[s1]) > 0.001:
            return 0
    return 1


while graphTest() is 0:
    iterateGraph()

# This code creates a list of all the similarity scores and ranks them in a max ordering
simList = []

sorted_list = sorted(newScoreMap.items(), key=operator.itemgetter(1))
sorted_list.reverse()
print(sorted_list)

sentenceNumber = math.floor(len(sorted_list) * 0.25)
print(sentenceNumber)
for x in range(sentenceNumber):
    print(sorted_list[x][0])

print()
print(len(sorted_list))
