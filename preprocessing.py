import gensim
import numpy as np
import os
import re

import nltk
from bs4 import BeautifulSoup
from rougescore import rougescore
from nltk.corpus import stopwords
from entry import Entry


class Preprocessing(object):
    doc_paths = ["duc2001_simplified/testing/"]
    sum_paths = ["duc2001_simplified/testing/summaries"]
    doc_paths_test = ["duc2002_simplified/data/"]
    sum_paths_test = ["duc2002_simplified/summaries"]
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    train_entries = None
    test_entries = None

    @staticmethod
    def get_train_summaries():
        summaries = []
        for filepath in Preprocessing.sum_paths:
            for dir in sorted(os.listdir(filepath)):
                for dir2 in os.listdir(filepath + "/" + dir):
                    if dir2 != "docs":
                        file = open(filepath + "/" + dir + "/" + dir2 + "/" + "perdocs")
                        text = file.read()
                        file.close()

                        # Extracting text from the XML files
                        soup = BeautifulSoup(text, 'html.parser')
                        documents = soup.findAll('sum')
                        for doc in documents:
                            docref = doc['docref'].upper()
                            doc = re.sub("<[^>]+>", '', str(doc)).replace('\n', ' ').replace('\r', '').lower()
                            summaries.append((docref, doc))

        return summaries

    @staticmethod
    def get_test_summaries():
        summaries = []
        for filepath in Preprocessing.sum_paths_test:
            for dir in sorted(os.listdir(filepath)):
                for sum in os.listdir(filepath + "/" + dir):
                    if sum == "perdocs":
                        file = open(filepath + "/" + dir + "/" + "perdocs")
                        text = file.read()
                        file.close()

                        # Extracting text from the XML files
                        soup = BeautifulSoup(text, 'html.parser')
                        documents = soup.findAll('sum')
                        for doc in documents:
                            docref = doc['docref'].upper()
                            doc = re.sub("<[^>]+>", '', str(doc)).replace('\n', ' ').replace('\r', '').lower()
                            summaries.append((docref, doc))
        return summaries

    @staticmethod
    def pad_sentence(sent, max_sent_len):
        diff = max_sent_len - len(sent)
        while diff > 0:
            diff -= 1
            sent.append("<PAD>")

        return sent

    @staticmethod
    def get_max_length(entries):

        # Loads words and pads
        tokenised_sentences = []
        for entry in entries:
             tokenised_sentences.append(entry.parsed_sentences)

        max_count = 0
        for sents in tokenised_sentences:
            for s in sents:
                if len(s) > max_count:
                    max_count = len(s)
        return max_count

    @staticmethod
    def get_sent_vectors(entries, max_sent_length):

        zero_vector = np.zeros(300)

        print("max sent length = ", max_sent_length)
        for entry in entries:
            sents = entry.parsed_sentences
            new_sents = []
            for s in sents:
                new_sents.append(Preprocessing.pad_sentence(s, max_sent_length))

            vectored_sents = []
            for sent in new_sents:
                word_vectors = []
                for word in sent:
                    try:
                        word_vectors.append(Preprocessing.model.get_vector(word))
                    except:
                        word_vectors.append(zero_vector)
                vectored_sents.append(word_vectors)
            entry.vectors = vectored_sents

        return entries

    @staticmethod
    def cut_sentences(train_entries, test_entries):
        new_train = []
        new_test = []
        for entry in train_entries:
            cut_sents = []
            for sent in entry.sentences:
                if len(sent) < 150:
                    cut_sents.append(sent)
            entry.sentences = cut_sents
            new_train.append(entry)
        for entry in test_entries:
            cut_sents = []
            for sent in entry.sentences:
                if len(sent) < 150:
                    cut_sents.append(sent)
            entry.sentences = cut_sents
            new_test.append(entry)

        return new_train, new_test

    @staticmethod
    def parse_sentences(train_entries, test_entries):
        stop_words = set(stopwords.words('english'))

        for e in train_entries:
            parsed_sents = []
            for s in e.sentences:
                words = nltk.word_tokenize(s)
                words = list(filter(lambda x: x not in stop_words, words))
                words = list(filter(lambda x: x is not '.', words))
                words = list(filter(lambda x: x is not ';', words))
                words = list(filter(lambda x: x is not ',', words))

                parsed_sents.append(words)
                e.parsed_sentences = parsed_sents

        for e in test_entries:
            parsed_sents = []
            for s in e.sentences:
                words = nltk.word_tokenize(s)
                words = list(filter(lambda x: x not in stop_words, words))
                words = list(filter(lambda x: x is not '.', words))
                words = list(filter(lambda x: x is not ';', words))
                words = list(filter(lambda x: x is not ',', words))

                parsed_sents.append(words)
                e.parsed_sentences = parsed_sents

        return train_entries, test_entries

    @staticmethod
    def read_files():

        train_entries = []
        test_entries = []
        train_summaries = Preprocessing.get_train_summaries()  # (docref, doc)
        test_summaries = Preprocessing.get_test_summaries()  # (docref, doc)

        # Reads in train files
        for filepath in Preprocessing.doc_paths:
            for filename in sorted(os.listdir(filepath)):
                if filename.endswith(".xml"):
                    dir = filename.replace(".xml", "")
                    file = open(filepath + filename)
                    text = file.read()
                    file.close()
                    # Extracting text from the XML files
                    soup = BeautifulSoup(text, 'html.parser')

                    documents = soup.findAll('doc')

                    for d in documents:
                        docref = str(d.find('docno')).replace("<docno>", "").replace("</docno>", "").replace("\n",
                                                                                                             "").upper()
                        texts = d.findAll('text')
                        text_set = []
                        for n in range(len(texts)):
                            text_set.append(
                                str(texts[n]).replace("<p>", " ").replace("</p>", " ").replace("<text>", " ") \
                                    .replace("</text>", " ").replace("\n", " "))

                        text = ''.join(text_set).lower()
                        sentences = nltk.sent_tokenize(text)

                        e = Entry()
                        e.dir = dir
                        e.doc = text
                        e.doc_id = docref
                        e.sentences = sentences
                        train_entries.append(e)

        # Reads in test files
        for filepath in Preprocessing.doc_paths_test:
            for filename in sorted(os.listdir(filepath)):
                if filename.endswith(".xml"):
                    dir = filename.replace(".xml", "")
                    file = open(filepath + filename)
                    text = file.read()
                    file.close()
                    # Extracting text from the XML files
                    soup = BeautifulSoup(text, 'html.parser')

                    documents = soup.findAll('doc')

                    for d in documents:
                        docref = str(d.find('docno')).replace("<docno>", "").replace("</docno>", "").replace(
                            "\n",
                            "").upper()
                        texts = d.findAll('text')
                        text_set = []
                        for n in range(len(texts)):
                            text_set.append(
                                str(texts[n]).replace("<p>", " ").replace("</p>", " ").replace("<text>", " ") \
                                    .replace("</text>", " ").replace("\n", " "))

                        text = ''.join(text_set).lower()
                        sentences = nltk.sent_tokenize(text)

                        e = Entry()
                        e.dir = dir
                        e.doc = text
                        e.doc_id = docref
                        e.sentences = sentences

                        test_entries.append(e)


        for entry in train_entries:
            for sum in train_summaries:
                if entry.doc_id == sum[0]:
                    entry.summary = sum[1]

        for entry in train_entries:
            if not isinstance(entry.summary, type("")):
                train_entries.remove(entry)

        for entry in test_entries:
            for sum in test_summaries:
                if entry.doc_id == sum[0]:
                    entry.summary = sum[1]

        for entry in test_entries:
            if not isinstance(entry.summary, type("")):
                test_entries.remove(entry)

        train_entries, test_entries = Preprocessing.cut_sentences(train_entries, test_entries)
        train_entries, test_entries = Preprocessing.parse_sentences(train_entries, test_entries)
        max_sent_length = Preprocessing.get_max_length(train_entries + test_entries)

        train_entries = Preprocessing.get_sent_vectors(train_entries, max_sent_length)
        test_entries = Preprocessing.get_sent_vectors(test_entries, max_sent_length)
        Preprocessing.train_entries = train_entries
        Preprocessing.test_entries = test_entries



    @staticmethod
    def get_salience_scores(alpha):

        stop_words = set(stopwords.words('english'))
        for entry in Preprocessing.train_entries:
            sentences = entry.parsed_sentences
            salience_scores = []
            summary_sents = nltk.word_tokenize(entry.summary)
            # Removing stop words
            summary_sents = list(filter(lambda x: x not in stop_words, summary_sents))
            summary_sents = list(filter(lambda x: x is not '.', summary_sents))
            summary_sents = list(filter(lambda x: x is not ';', summary_sents))
            summary_sents = list(filter(lambda x: x is not ',', summary_sents))

            for input_sent in sentences:
                rouge1 = rougescore.rouge_1(input_sent, [summary_sents], 0.5)
                rouge2 = rougescore.rouge_2(input_sent, [summary_sents], 0.5)

                salience_score = alpha * rouge1 + (1 - alpha) * rouge2
                salience_scores.append(salience_score)

            entry.saliences = salience_scores

        for entry in Preprocessing.test_entries:
            sentences = entry.parsed_sentences
            salience_scores = []

            summary_sents = nltk.word_tokenize(entry.summary)
            # Removing stop words
            summary_sents = list(filter(lambda x: x not in stop_words, summary_sents))
            summary_sents = list(filter(lambda x: x is not '.', summary_sents))
            summary_sents = list(filter(lambda x: x is not ';', summary_sents))
            summary_sents = list(filter(lambda x: x is not ',', summary_sents))

            for input_sent in sentences:

                rouge1 = rougescore.rouge_1(input_sent, [summary_sents], 0.5)
                rouge2 = rougescore.rouge_2(input_sent, [summary_sents], 0.5)

                salience_score = alpha * rouge1 + (1 - alpha) * rouge2

                salience_scores.append(salience_score)

            entry.saliences = salience_scores


    @staticmethod
    def get_cnn_vectors():
        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for entry in Preprocessing.train_entries:
            for x in range(len(entry.vectors)):
                data_val = entry.vectors[x]
                label_val = entry.saliences[x]
                train_data.append(data_val)
                train_labels.append(label_val)

        train_data = np.asarray(train_data, dtype=np.float32)
        train_labels = np.asarray(train_labels, dtype=np.float32)

        train_data = np.expand_dims(train_data, axis=3)
        train_labels = np.expand_dims(train_labels, axis=1)

        for entry in Preprocessing.test_entries:
            for x in range(len(entry.vectors)):
                data_val = entry.vectors[x]
                label_val = entry.saliences[x]
                test_data.append(data_val)
                test_labels.append(label_val)

        test_data = np.asarray(test_data, dtype=np.float32)
        test_labels = np.asarray(test_labels, dtype=np.float32)

        test_data = np.expand_dims(test_data, axis=3)
        test_labels = np.expand_dims(test_labels, axis=1)
        return train_data, train_labels, test_data, test_labels
