import gensim
import numpy as np
import os
import re
import copy

import nltk
from bs4 import BeautifulSoup
from pyrouge import Rouge155
from rougescore import rougescore
from nltk.corpus import stopwords
from entry import Entry
from nltk.stem import PorterStemmer


class Preprocessing(object):
    doc_paths = ["duc2001_simplified/testing/"]
    sum_paths = ["duc2001_simplified/testing/summaries"]
    doc_paths_test = ["duc2002_simplified/data/"]
    sum_paths_test = ["duc2002_simplified/summaries"]
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

    def __init__(self, stem):
        self.stem = stem
        self.train_entries = None
        self.test_entries = None

    @staticmethod
    def get_train_summaries():
        # Gets the relevant summary to the train sentenes

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
        # Aquires the relevant summary to the test sentence

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
        new_sent = copy.copy(sent)
        diff = max_sent_len - len(new_sent)
        while diff > 0:
            diff -= 1
            new_sent.append("<PAD>")

        return new_sent

    @staticmethod
    def get_max_length(entries):
        # Finds the token length of the longest sentence

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
    def get_sent_vectors(entries, max_sent_length, rand):
        # Uses Word2Vec to get list of sentence vectors from words

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
                    if not rand:
                        try:
                            word_vectors.append(Preprocessing.model.get_vector(word))
                        except:
                            word_vectors.append(zero_vector)
                    else:
                        word_vectors.append(np.random.random(300))
                vectored_sents.append(word_vectors)
            entry.vectors = vectored_sents
        return entries

    @staticmethod
    def cut_sentences(train_entries, test_entries):
        # Cuts sentences larger than 150 long to filter out anomalies

        new_train = []
        new_test = []
        for entry in train_entries:
            cut_sents = []
            for sent in entry.sentences:
                if len(nltk.word_tokenize(sent)) < 150:
                    cut_sents.append(sent)
            entry.sentences = cut_sents
            new_train.append(entry)
        for entry in test_entries:
            cut_sents = []
            for sent in entry.sentences:
                if len(nltk.word_tokenize(sent)) < 150:
                    cut_sents.append(sent)
            entry.sentences = cut_sents
            new_test.append(entry)

        return new_train, new_test

    def parse_sentences(self, train_entries, test_entries):
        # Filters sentences to include useful words only

        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()

        for e in train_entries + test_entries:
            parsed_sents = []
            for s in e.sentences:
                words = nltk.word_tokenize(s)
                words = list(filter(lambda x: x not in stop_words, words))
                words = list(filter(lambda x: x is not '.', words))
                words = list(filter(lambda x: x is not ';', words))
                words = list(filter(lambda x: x is not ',', words))
                if self.stem:
                    words = list(map(lambda x: ps.stem(x), words))

                parsed_sents.append(words)

            e.parsed_sentences = parsed_sents

        return train_entries, test_entries

    def read_files(self):
        # Reads from documents to process into files.

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
        self.parse_sentences(train_entries, test_entries)
        max_sent_length = Preprocessing.get_max_length(train_entries + test_entries)

        train_entries = Preprocessing.get_sent_vectors(train_entries, max_sent_length, False)
        test_entries = Preprocessing.get_sent_vectors(test_entries, max_sent_length, False)
        self.train_entries = train_entries
        self.test_entries = test_entries

    def get_salience_scores(self, alpha):
        # Calculates the similarity between each sentence and their respective training summary

        # Clearing folders
        # list(map(os.unlink, (os.path.join("model_summaries", f) for f in os.listdir("model_summaries"))))
        # list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))
        #
        # for x in range(len(self.train_entries)):
        #     entry = self.train_entries[x]
        #     model_sum = entry.summary
        #     sent_scores = []
        #
        #     sentences = nltk.sent_tokenize(model_sum)
        #     file = open("model_summaries/model_sum.A."+str(0)+".txt", "w+")
        #     for s in sentences:
        #         file.write(s+"\n")
        #     file.close()
        #
        #     sentences = entry.sentences
        #     for s in sentences:
        #         file = open("system_summaries/system_sum."+str(0)+".txt", "w+")
        #         file.write(s)
        #         file.close()
        #
        #         r = Rouge155("ROUGE-1.5.5",
        #                      rouge_args="-e ROUGE-1.5.5/data -a -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0")
        #
        #         r.system_dir = 'system_summaries'
        #         r.model_dir = 'model_summaries'
        #         r.system_filename_pattern = 'system_sum.(\d+).txt'
        #         r.model_filename_pattern = 'model_sum.[A-Z].#ID#.txt'
        #
        #         output = r.convert_and_evaluate()
        #         output_dict = r.output_to_dict(output)
        #         score = (output_dict['rouge_1_f_score'] + output_dict['rouge_2_f_score'])/2
        #         print(score)
        #         sent_scores.append(score)
        #         # Clearing folder
        #         list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))
        #
        #     list(map(os.unlink, (os.path.join("model_summaries", f) for f in os.listdir("model_summaries"))))
        #     print(sent_scores)
        #     entry.saliences = sent_scores

        stop_words = set(stopwords.words('english'))

        for entry in self.train_entries:
            sentences = entry.parsed_sentences
            salience_scores = []

            if self.stem:
                ps = PorterStemmer()
                summary_sents = [ps.stem(x) for x in nltk.word_tokenize(entry.summary)]
            else:
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
                print(salience_score)

            entry.saliences = salience_scores

    def get_cnn_vectors(self, train):
        # Gets the training vectors in a useful manner to input into the CNN

        train_data = []
        train_labels = []
        test_data = []

        if train:
            for entry in self.train_entries:
                for x in range(len(entry.vectors)):
                    data_val = entry.vectors[x]
                    label_val = entry.saliences[x]
                    train_data.append(data_val)
                    train_labels.append(label_val)

            print(np.shape(train_data))
            print(np.shape(train_labels))

            train_data = np.asarray(train_data, dtype=np.float32)
            train_labels = np.asarray(train_labels, dtype=np.float32)

            train_data = np.expand_dims(train_data, axis=3)
            train_labels = np.expand_dims(train_labels, axis=1)

        else:
            for entry in self.test_entries:
                for x in range(len(entry.vectors)):
                    data_val = entry.vectors[x]
                    test_data.append(data_val)

            test_data = np.asarray(list(test_data), dtype=np.float32)
            test_data = np.expand_dims(test_data, axis=3)

        return train_data, train_labels, test_data
