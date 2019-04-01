import nltk
import numpy as np
from random import random, shuffle

from postprocessing import Postprocess
from preprocessing import Preprocessing
from text_cnn import TextCNN
from extension import Extension
import textrank


class Main(object):

    def __init__(self):
        self.pre = None
        self.post = None

    def load_dataset(self, stem):
        # Runs tests depending on parameters
        print("initialising training models...")

        self.pre = Preprocessing(stem)
        self.post = Postprocess(self.pre)

        print("reading files...")

        self.pre.read_files()

    def text_rank(self):
        test_entries = self.pre.test_entries
        textrank.TextRank.run(test_entries)

    def extension(self):
        test_entries = self.pre.test_entries
        Extension(test_entries)

    def control_case(self):
        test_entries = self.pre.test_entries

        def get_summary_sentences(word_count, list):

            count = word_count
            summary = ""
            for sent in list:
                length = len(nltk.word_tokenize(sent))
                if count < 0:
                    break
                else:
                    index = None
                    for i in range(len(sent)):
                        if sent[i].isalnum():
                            index = i
                            break
                    new_sent = sent[index:]
                    summary += new_sent + " "
                    count -= length

            return summary

        for e in test_entries:
            sentences = e.sentences
            new_sentences = sentences[:]  # Copy sentences
            shuffle(new_sentences)
            summary = get_summary_sentences(100, new_sentences)
            e.control_sum = summary

    @staticmethod
    def digit_count(entries):
        cnn_dig_count = 0
        tr_dig_count = 0
        for entry in entries:
            cnn_dig_count += sum(c.isdigit() for c in entry.generated_sum)
            tr_dig_count += sum(c.isdigit() for c in entry.text_rank_sum)
        print("CNN dig count= ", cnn_dig_count)
        print("TextRank dig count= ", tr_dig_count)

    def main(self):
        print("loading datasets...")
        self.load_dataset(stem=False)
        print("running control case model...")
        self.control_case()
        print("running TextRank model...")
        self.text_rank()
        print("running CNN model...")
        self.cnn(train=False)
        print("running extension")
        self.extension()

        print("calculating rouge scores...")

        self.post.calculate_rouge(0.5)
        self.digit_count(self.pre.test_entries)

        # print(self.pre.test_entries[0].control_sum)
        # print(self.pre.test_entries[0].text_rank_sum)
        # print(self.pre.test_entries[0].generated_sum)
        # print(self.pre.test_entries[0].summary)
        # print(self.pre.test_entries[0].sentences)

    def cnn(self, train):

        if train:
            print("getting initial salience scores...")
            self.pre.get_salience_scores(load=True)

            train_data, train_labels, _ = self.pre.get_cnn_vectors(train)

            # We've now pre process the documents, so now we can feed into the CNN

            TextCNN(train_data=train_data, train_labels=train_labels, num_filters=400, kernel_size=3)
            exit()
        else:
            _, _, test_data = self.pre.get_cnn_vectors(train)

            print("generating salience results...")

            gen_salience_results = TextCNN.eval(test_data)
            print(gen_salience_results)

            print("post-processing results...")
            self.post.get_results(gen_salience_results, test_data)

            print("getting summaries...")

            self.post.get_summary_sentences(100)
            self.post.return_summaries()


if __name__ == '__main__':
    Main().main()
