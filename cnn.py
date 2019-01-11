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
        entries = Preprocessing.train_entries
        for e in entries:
            results = []
            for sent in e.sentences:
                results.append((sent, salience_scores.pop(0)[0]))
            e.output = sorted(results, key=lambda y: y[1], reverse=True)

        Preprocessing.train_entries = entries

    @staticmethod
    def calculate_rouge(selected_summary_map):
        pass

def main():

    train_filepaths = ["duc2001_simplified/testing/"]#, "duc2001_simplified/training/"]
    test_filepaths = ["duc2002_simplified/data/"]

    train_summary_paths = ["duc2001_simplified/testing/summaries"]
    test_summary_paths = ["duc2002_simplified/summaries"]


    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                            binary=True)

    Preprocessing.read_files()
    Preprocessing.get_salience_scores(0.5)

    train_data, train_labels = Preprocessing.get_cnn_vectors()

    # We've now pre process the documents, so now we can feed into the CNN

    #cnn = TextCNN(train_data=train_data, train_labels=train_labels, num_filters=400, kernel_size=3)

    salience_results = TextCNN.eval(train_data, train_labels)
    Postprocess.get_results(salience_results)

    for e in Preprocessing.train_entries:
        print(e.output)


    """
    doc_sent_score_map = Postprocess.get_results(salience_results, prep_test.get_doc_sentence_map())
    generated_summary_map = Postprocess.select_sentences(doc_sent_score_map)
    Postprocess.save_generated_summaries(generated_summary_map)
    """

if __name__ == '__main__':
    main()

