import datetime
import time

import gensim
import math
import nltk
import rougescore
import xml.etree.ElementTree as ET
import tensorflow as tf
from bs4 import BeautifulSoup
import numpy as np
import logging
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
import os
import copy
from text_cnn import TextCNN

from tensorflow.python.platform.flags import FLAGS


class Preprocess(object):

    def __init__(
            self, filepaths):
        self.filepaths = filepaths


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

    # Creating map of each document with its sentences Returns MAP<DOCS, [SENTENCES]>
    def get_doc_sentence_map(self):

        doc_catalog = []
        for filepath in self.filepaths:
            for filename in os.listdir(filepath):
                if filename.endswith(".xml"):
                    file = open(filepath + filename)
                    text = file.read()
                    file.close()

                    # Extracting text from the XML files
                    soup = BeautifulSoup(text, 'html.parser')

                    documents = soup.findAll('text')
                    new_set = []
                    for doc in range(len(documents)):
                        new_set.append(str(documents[doc]).replace("<p>", " ").replace("</p>", " ").replace("<text>", " ") \
                                       .replace("</text>", " ").replace("\n", " "))

                    doc_catalog.append(new_set)

        train_reviews = {}
        for doc in doc_catalog:
            for text in doc:
                text_sentences = nltk.sent_tokenize(text)
                train_reviews[text] = text_sentences

        return train_reviews

    # Creating map of each document with its sentences represented as vectors Returns MAP<STRING, [(SENT, VECTOR)]>
    def create_word_vectors(self, model):
        zero_vector = np.zeros(300)
        total_sent_vecs = {}
        doc_sent_map = self.get_doc_sentence_map()  # MAP<DOCS, [SENTENCES]>
        doc_sent_map_copy = copy.deepcopy(doc_sent_map)
        for doc in doc_sent_map:
            for s in range(len(doc_sent_map[doc])):
                words = nltk.word_tokenize(doc_sent_map[doc][s])
                doc_sent_map[doc][s] = words

        doc_sent_map = self.pad_sentences(doc_sent_map)

        for doc in doc_sent_map:
            total_sent_vecs[doc] = []
            sents = doc_sent_map[doc]
            for s in range(len(sents)):
                word_vecs = []
                for word in sents[s]:
                    try:
                        word_vector = model.get_vector(word)
                    except:
                        word_vector = zero_vector

                    word_vecs.append(word_vector)

                total_sent_vecs[doc].append((doc_sent_map_copy[doc][s], word_vecs))

        return total_sent_vecs

    # Calculating sentence salience score using Rouge metric. returns MAP<DOC,[(SENT, SCORE)]>
    @staticmethod
    def get_salience_scores(doc_sent_map, alpha):
        sentence_salience_score_map = {}
        sent_map = copy.deepcopy(doc_sent_map)
        for doc in doc_sent_map:
            sentence_salience_score_map[doc] = []

            for sent in range(len(doc_sent_map[doc])):
                doc_sent_map[doc][sent] = nltk.word_tokenize(doc_sent_map[doc][sent])

            for sent in range(len(doc_sent_map[doc])):
                model_sentences = copy.deepcopy(doc_sent_map[doc])
                input_sent = model_sentences[sent]
                del (model_sentences[sent])
                rouge1 = rougescore.rouge_1(input_sent, model_sentences, 0.5)
                rouge2 = rougescore.rouge_2(input_sent, model_sentences, 0.5)

                salience_score = alpha * rouge1 + (1 - alpha) * rouge2

                sentence_salience_score_map[doc].append((sent_map[doc][sent], salience_score))
        return sentence_salience_score_map

    # Here we start the tensorflow implementation, we need all Word2Vec vectors stored so we can input into the model
    # 1 vector for each document I think

    # Fills doc -> [sentences] with <PAD> token so sentence matrices are same size
    @staticmethod
    def pad_sentences(sentence_tokens):
        max_token_count = 0

        for doc in sentence_tokens:
            for sent in sentence_tokens[doc]:
                if len(sent) > max_token_count:
                    max_token_count = len(sent)

        for doc in sentence_tokens:
            for sent in sentence_tokens[doc]:
                while len(sent) < max_token_count:
                    sent.append('<PAD>')

        return sentence_tokens

    # This returns the MAP<DOC, [(VECTOR, SCORE)]>
    def get_vec_salience_map(self, model):

        doc_sentence_vectors = self.create_word_vectors(
            model)  # MAP<DOC, [(SENT, VECTOR)]> for doc, for array, for tuple

        salience_scores = self.get_salience_scores(self.get_doc_sentence_map(),
                                              0.5)  # MAP<DOC,[(SENT, SCORE)]>  for doc, for tuple

        doc_sent_vec_salience_map = {}

        for doc in salience_scores:
            doc_sent_vec_salience_map[doc] = []
            for tuple1 in doc_sentence_vectors[doc]:
                for tuple2 in salience_scores[doc]:
                    if tuple1[0] == tuple2[0]:
                        doc_sent_vec_salience_map[doc].append((tuple1[1], tuple2[1]))

        return doc_sent_vec_salience_map


def main():

    train_filepaths = ["duc2001_simplified/training/", "duc2001_simplified/testing/"]
    test_filepaths = ["duc2002_simplified/data/"]

    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    prep_train = Preprocess(filepaths=train_filepaths)
    train_vec_sal_map = Preprocess.get_vec_salience_map(prep_train, model)
    prep_test = Preprocess(filepaths=test_filepaths)
    test_vec_sal_map = Preprocess.get_vec_salience_map(prep_test, model)
    print(len(train_vec_sal_map))
    print(len(test_vec_sal_map))

    train_data = []
    train_labels = []
    for doc in train_vec_sal_map:
        for sent in train_vec_sal_map[doc]:
            train_data.append(sent[0])
            train_labels.append(sent[1])

    for x in range(0, 10):
        print(len(train_data[x]))
    print(len(train_data))

    test_data = []
    test_labels = []
    for doc in test_vec_sal_map:
        for sent in test_vec_sal_map[doc]:
            test_data.append(sent[0])
            test_labels.append(sent[1])
    for y in range(0,10):
        print(len(test_data[y]))
    print(len(test_data))
    exit()
    train_data = np.asarray(train_data, dtype=np.float16)
    train_labels = np.asarray(train_labels, dtype=np.float16)
    test_data = np.asarray(test_data, dtype=np.float16)
    test_labels = np.asarray(test_labels, dtype=np.float16)


    # We've now pre process the documents, so now we can feed into the CNN

    cnn = TextCNN(train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels,
                  num_filters=300, kernel_size=3)


if __name__ == '__main__':
    main()

