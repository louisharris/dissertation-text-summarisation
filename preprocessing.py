import gensim
import numpy as np
import os
import re

import nltk
from bs4 import BeautifulSoup
from rougescore import rougescore

from entry import Entry


class Preprocessing(object):
    doc_paths = ["duc2001_simplified/testing/"]
    sum_paths = ["duc2001_simplified/testing/summaries"]
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                            binary=True)
    train_entries = None
    test_entries = None

    @staticmethod
    def get_summaries():
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
                            doc = re.sub("<[^>]+>", '', str(doc)).replace('\n', ' ').replace('\r', '')
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
    def get_max_length(sentences):
        max_count = 0
        for sent in sentences:
            if len(sent) > max_count:
                max_count = len(sent)
        return max_count

    # Creating map of each document with its sentences represented as vectors Returns MAP<STRING, [(SENT, VECTOR)]>
    @staticmethod
    def get_sent_vectors(entries):

        # Loads words and pads
        zero_vector = np.zeros(300)
        tokenised_sentences = []
        for entry in entries:
            sents = entry.sentences
            for s in sents:
                tokenised_sentences.append(nltk.word_tokenize(s))

        max_sent_length = Preprocessing.get_max_length(tokenised_sentences)
        print("max sent length = ", max_sent_length)

        for entry in entries:
            sents = entry.sentences
            new_sents = []
            for s in sents:
                s = nltk.word_tokenize(s)
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
    def read_files():

        entries = []
        summaries = Preprocessing.get_summaries()  # (docref, doc)

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

                        text = ''.join(text_set)

                        e = Entry()
                        e.dir = dir
                        e.doc = text
                        e.doc_id = docref
                        e.sentences = nltk.sent_tokenize(text)
                        entries.append(e)

        for entry in entries:
            for sum in summaries:
                if entry.doc_id == sum[0]:
                    entry.summary = sum[1]

        for entry in entries:
            if not isinstance(entry.summary, type("")):
                entries.remove(entry)

        entries = Preprocessing.get_sent_vectors(entries)
        Preprocessing.train_entries = entries

    @staticmethod
    def get_salience_scores(alpha):
        for entry in Preprocessing.train_entries:
            sentences = entry.sentences
            salience_scores = []
            for s in sentences:
                input_sent = nltk.word_tokenize(s)
                summary_sents = nltk.word_tokenize(entry.summary)

                rouge1 = rougescore.rouge_1(input_sent, [summary_sents], 0.5)
                rouge2 = rougescore.rouge_2(input_sent, [summary_sents], 0.5)

                salience_score = alpha * rouge1 + (1 - alpha) * rouge2

                salience_scores.append(salience_score)

            entry.saliences = salience_scores

    @staticmethod
    def get_cnn_vectors():
        train_data = []
        train_labels = []

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

        return train_data, train_labels
