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


class Preprocess(object):

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

    @staticmethod
    def get_results(salience_scores):

        salience_scores = list(salience_scores)
        entries = Preprocessing.test_entries

        print(len(salience_scores))
        for e in entries:
            results = []
            for sent in e.sentences:
                results.append((sent, salience_scores.pop(0)[0]))
            e.output = sorted(results, key=lambda y: y[1], reverse=True)


        Preprocessing.test_entries = entries

    @staticmethod
    def calculate_rouge(alpha):
        for entry in Preprocessing.test_entries:
            original_sum = nltk.word_tokenize(entry.summary)
            generated_sum = nltk.word_tokenize(entry.generated_sum)

            rouge1 = rougescore.rouge_1(generated_sum, [original_sum], alpha)
            rouge2 = rougescore.rouge_2(generated_sum, [original_sum], alpha)

            entry.rouge_scores = (rouge1, rouge2)

    @staticmethod
    def calculate_rouge_alt():
        for entry in Preprocessing.train_entries:
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


    @staticmethod
    def get_summary_sentences(word_count):

        for entry in Preprocessing.test_entries:
            sentences = entry.output
            count = word_count
            new_output = []
            for tuple in sentences:
                length = len(nltk.word_tokenize(tuple[0]))
                if count < 0:
                    break
                else:
                    new_output.append(tuple)
                    count -= length

            entry.output = new_output

    @staticmethod
    def return_summaries():
        for entry in Preprocessing.test_entries:
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


def main():

    train_filepaths = ["duc2001_simplified/testing/"]#, "duc2001_simplified/training/"]
    test_filepaths = ["duc2002_simplified/data/"]

    train_summary_paths = ["duc2001_simplified/testing/summaries"]
    test_summary_paths = ["duc2002_simplified/summaries"]

    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                            binary=True)

    Preprocessing.read_files()
    Preprocessing.get_salience_scores(0.5)

    train_data, train_labels, test_data, test_labels = Preprocessing.get_cnn_vectors()
    # We've now pre process the documents, so now we can feed into the CNN

    print(len(test_data))
    print(len(test_labels))



    #cnn = TextCNN(train_data=train_data, train_labels=train_labels, num_filters=400, kernel_size=3)
    #exit()
    salience_results = TextCNN.eval(test_data, test_labels)
    print(salience_results)

    Postprocess.get_results(salience_results)
    Postprocess.get_summary_sentences(100)
    Postprocess.return_summaries()
    Postprocess.calculate_rouge(0.5)

    #for x in range(len(Preprocessing.train_entries)):
        #print(Preprocessing.test_entries[x].generated_sum)
        #print()


    mean_rouge = []
    for entry in Preprocessing.test_entries:
        mean_rouge.append(entry.rouge_scores)
    print(mean_rouge)
    print(np.mean(mean_rouge, axis=0))


if __name__ == '__main__':
    main()

