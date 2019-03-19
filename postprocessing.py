import datetime
import time

import gensim
import math
import nltk
import rougescore
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import numpy as np
import logging
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
import os
import copy
from text_cnn import TextCNN
import re
from entry import Entry
import tensorflow as tf
from preprocessing import Preprocessing
from nltk.stem import PorterStemmer



class TrainWord2vec(object):

    @staticmethod
    def train_word2_vec(parameters, documents):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        doc_list = []
        for doc in documents:
            tokens = nltk.word_tokenize(doc)
            doc_list.append(tokens)

        model = Word2Vec(doc_list, size=100, window=5, min_count=0, workers=4)
        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        model.save('./word2vec_model.d2v')
        return model

    # Here we start the tensorflow implementation, we need all Word2Vec vectors stored so we can input into the model
    # 1 vector for each document I think


class Postprocess(object):

    def __init__(self, pre):
        self.pre = pre

    def get_results(self, salience_scores):

        salience_scores = list(salience_scores)
        entries = self.pre.test_entries

        for e in entries:
            results = []
            for sent in e.sentences:
                results.append((sent, salience_scores.pop(0)[0]))
            e.output = sorted(results, key=lambda y: y[1], reverse=True)

        self.pre.test_entries = entries

    def calculate_rouge(self, alpha):
        ps = PorterStemmer()

        for entry in self.pre.test_entries:

            original_sum = nltk.word_tokenize(entry.summary)
            original_sum_stemmed = [ps.stem(x) for x in original_sum]

            # Generate ROUGE scores for CNN

            generated_sum = nltk.word_tokenize(entry.generated_sum)
            generated_sum_stemmed = [ps.stem(x) for x in generated_sum]

            rouge1 = rougescore.rouge_1(generated_sum_stemmed, [original_sum_stemmed], alpha)
            rouge2 = rougescore.rouge_2(generated_sum_stemmed, [original_sum_stemmed], alpha)

            entry.rouge_scores_cnn = (rouge1, rouge2)

            # Generate ROUGE scores for TextRank
            generated_sum = nltk.word_tokenize(entry.text_rank_sum)
            generated_sum_stemmed = [ps.stem(x) for x in generated_sum]

            rouge1 = rougescore.rouge_1(generated_sum_stemmed, [original_sum_stemmed], alpha)
            rouge2 = rougescore.rouge_2(generated_sum_stemmed, [original_sum_stemmed], alpha)

            entry.rouge_scores_tr = (rouge1, rouge2)

            # Generate ROUGE scores for control case
            generated_sum = nltk.word_tokenize(entry.control_sum)
            generated_sum_stemmed = [ps.stem(x) for x in generated_sum]

            rouge1 = rougescore.rouge_1(generated_sum_stemmed, [original_sum_stemmed], alpha)
            rouge2 = rougescore.rouge_2(generated_sum_stemmed, [original_sum_stemmed], alpha)

            entry.rouge_scores_control = (rouge1, rouge2)

    def calculate_rouge_alt(self):
        for entry in self.pre.train_entries:
            original_sum = nltk.word_tokenize(entry.summary)
            generated_sum = nltk.word_tokenize(entry.generated_sum)

            bigram_orig = list(nltk.bigrams(original_sum))
            bigram_gen = list(nltk.bigrams(generated_sum))

            count = 0
            for word in original_sum:
                for w in generated_sum:
                    if word == w:
                        count += 1
                        break
            recall = count / len(original_sum)
            precision = count / len(generated_sum)

            count = 0
            for word in bigram_orig:
                for w in bigram_gen:
                    if word == w:
                        count += 1
                        break
            recall_big = count / len(bigram_orig)
            precision_big = count / len(bigram_gen)

            entry.rouge_scores = ((recall + precision) / 2, (recall_big + precision_big) / 2)

    def get_summary_sentences(self, word_count):

        for entry in self.pre.test_entries:
            sentences = entry.output
            count = word_count
            new_output = []
            for tup in sentences:
                length = len(nltk.word_tokenize(tup[0]))
                if count < 0:
                    break
                else:
                    new_output.append(tup)
                    count -= length

            entry.output = new_output

    def return_summaries(self):
        for entry in self.pre.test_entries:
            summary = ""
            for tuple in entry.output:
                if re.search('[a-zA-Z0-9]', tuple[0]) is not None:
                    index = None
                    for i in range(len(tuple[0])):
                        if tuple[0][i].isalnum():
                            index = i
                            break
                    new_sent = tuple[0][index:]
                    summary += new_sent + " "
            entry.generated_sum = summary


