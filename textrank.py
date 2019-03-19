# This is my implementation of the TextRank algorithm to generate summaries from documents.
import math
import operator
import random
import nltk


class TextRank(object):

    @staticmethod
    def similarity(s1, s2):
        count = 1
        # tokens_s1 = nltk.word_tokenize(s1)
        # tokens_s2 = nltk.word_tokenize(s2)
        for w1 in s1:
            for w2 in s2:
                if w1 == w2:
                    count += 1

        len1 = len(s1)
        len2 = len(s2)

        if len1 <= 1:
            len1 += 1
        if len2 <= 1:
            len2 += 1

        sim_score = count / (math.log(len1) + math.log(len2))
        return sim_score

    @staticmethod
    def get_summary_sentences(word_count, sorted_list):

        count = word_count
        summary = ""
        for tup in sorted_list:
            length = len(nltk.word_tokenize(tup[0]))
            if count < 0:
                break
            else:
                summary += tup[0]
                count -= length

        return summary

    @staticmethod
    def run(entries):

        # with open('data.txt', 'r') as myfile:
        #     data = myfile.read().replace('\n', '')

        # sentences = nltk.tokenize.sent_tokenize(data)
        # print(sentences)

        # This code creates a similarity score mapping between each sentence and the other

        for e in entries:
            sim_map = {}

            sent_dict = dict(zip(e.sentences, e.parsed_sentences))
            for s1 in e.sentences:
                sim_map[s1] = {}

            for s1 in e.sentences:
                for s2 in e.sentences:
                    if s1 != s2:
                        sim_score = TextRank.similarity(sent_dict[s1], sent_dict[s2])
                        sim_map[s1][s2] = sim_score
            # print(sim_map)
            # print(len(sim_map))

            # This code uses the similarity score mappings to iteratively calculate the weighted score of each sentence

            new_score_map = {}
            old_score_map = {}

            for s1 in e.sentences:
                new_score_map[s1] = random.uniform(0, 10)
            for s1 in e.sentences:
                old_score_map[s1] = 0

            def graph_test():
                for s1 in e.sentences:
                    if abs(old_score_map[s1] - new_score_map[s1]) > 0.001:
                        return 0
                return 1

            def iterate_graph():
                for S1 in e.sentences:
                    out_sum = 0
                    in_sum = 0
                    for S2 in sim_map[S1]:
                        out_sum = out_sum + sim_map[S1][S2]
                    for S2 in sim_map[S1]:
                        in_sum = in_sum + (sim_map[S2][S1] / out_sum) * (new_score_map[S2])
                    new_score = (1 - 0.85) + 0.85 * in_sum
                    old_score_map[S1] = new_score_map[S1]
                    new_score_map[S1] = new_score

            while graph_test() is 0:
                iterate_graph()

            # This code creates a list of all the similarity scores and ranks them in a max ordering
            sorted_list = sorted(new_score_map.items(), key=operator.itemgetter(1))
            sorted_list.reverse()

            summary = TextRank.get_summary_sentences(100, sorted_list)
            e.text_rank_sum = summary
