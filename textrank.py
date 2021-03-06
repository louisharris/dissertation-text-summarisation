import math
import numpy as np
import operator
import random
import nltk
import gensim

from scipy import spatial
from preprocessing import Preprocessing

model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
text_vec_map = {}


class TextRank(object):
    """
    Implementation of the TextRank algorithm to generate summaries from documents.
    """

    @staticmethod
    def similarity(s1, s2):
        # Bag-of-words method to calculate similarity scores between 2 sentences

        count = 1
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

        assert(sim_score >= 0)

        return sim_score

    @staticmethod
    def word_2_vec_similarity(s1, s2):
        # Cosine similarity method to calculate similarity between
        # two sentences

        s1_string = "".join(s1)
        s2_string = "".join(s2)

        if s1_string in text_vec_map:
            avg_s1 = text_vec_map[s1_string]
        else:
            new_s1 = Preprocessing.pad_sentence(s1, 94)
            s1_vectors = []
            for word in new_s1:
                try:
                    s1_vectors.append(Preprocessing.model.get_vector(word))
                except:
                    vec = np.random.uniform(-0.1, 0.1, 300)
                    s1_vectors.append(vec)
            avg_s1 = np.mean(s1_vectors, axis=0)
            text_vec_map[s1_string] = avg_s1
        if s2_string in text_vec_map:
            avg_s2 = text_vec_map[s2_string]
        else:
            new_s2 = Preprocessing.pad_sentence(s2, 94)
            s2_vectors = []

            for word in new_s2:
                try:
                    s2_vectors.append(Preprocessing.model.get_vector(word))
                except:
                    vec = np.random.uniform(-0.1, 0.1, 300)
                    s2_vectors.append(vec)
            avg_s2 = np.mean(s2_vectors, axis=0)
            text_vec_map[s2_string] = avg_s2

        similarity = 1 - spatial.distance.cosine(avg_s1, avg_s2)

        assert(0 <= similarity <= 1)

        return similarity

    @staticmethod
    def get_summary_sentences(word_count, sorted_list):
        # Obtains summary sentences filling the word count

        count = word_count
        summary = ""
        sents = [tup[0] for tup in sorted_list]
        for sent in sents:
            if count < 0:
                break
            else:
                index = None
                for i in range(len(sent)):
                    if sent[i].isalnum():
                        index = i
                        break
                new_sent = sent[index:]
                length = len(nltk.word_tokenize(new_sent))
                summary += new_sent + " "
                count -= length

        assert(count >= 0 or len(nltk.word_tokenize(summary)) >= 100)
        return summary

    @staticmethod
    def run(entries):
        # This code creates a similarity score mapping between each sentence and the other

        iteration_counts = []
        for e in entries:
            sim_map = {}

            sent_dict = dict(zip(e.sentences, e.parsed_sentences))
            for s1 in e.sentences:
                sim_map[s1] = {}

            for s1 in e.sentences:
                for s2 in e.sentences:
                    if s1 != s2:
                        sim_score = TextRank.similarity(sent_dict[s1], sent_dict[s2])
                        # sim_score = TextRank.word_2_vec_similarity(sent_dict[s1], sent_dict[s2])
                        sim_map[s1][s2] = sim_score

            # Uses similarity score mappings to iteratively calculate the
            # weighted score of each sentence
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

            # Iteration
            iterate_count = 0
            while graph_test() is 0:
                iterate_graph()
                iterate_count += 1
            iteration_counts.append(iterate_count)

            # Creates a list of all the similarity scores and ranks them in a max ordering
            sorted_list = sorted(new_score_map.items(), key=operator.itemgetter(1), reverse=True)
            e.output_tr = sorted_list

            summary = TextRank.get_summary_sentences(100, sorted_list)

            assert(summary is not None)

            e.text_rank_sum = summary
        print("Converged after ", np.mean(iteration_counts), " iterations")
