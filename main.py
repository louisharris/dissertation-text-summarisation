import numpy as np
import matplotlib.pyplot as plt
import nltk
import textrank

from random import shuffle
from postprocessing import Postprocess
from preprocessing import Preprocessing
from text_cnn import TextCNN
from extension import Extension
from plotting import Plotting


class Main(object):
    """
    Runs the main functions of the system in order,
    collecting the summary results.
    """

    def __init__(self):
        self.pre = None
        self.post = None

    def load_dataset(self, stem):
        # Loads and reads in data files

        print("initialising training models...")

        self.pre = Preprocessing(stem)
        self.post = Postprocess(self.pre)

        print("reading files...")

        self.pre.read_files()

        assert(self.pre is not None)
        assert(self.post is not None)

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

            assert(len(nltk.word_tokenize(summary)) >= word_count)
            return summary

        for e in test_entries:
            sentences = e.sentences
            new_sentences = sentences[:]  # Copy sentences
            shuffle(new_sentences)
            summary = get_summary_sentences(100, new_sentences)
            assert(summary is not None)
            e.control_sum = summary

    @staticmethod
    def digit_count(entries):
        # Counts number of digits present in all entries

        cnn_dig_count = 0
        tr_dig_count = 0
        for entry in entries:
            cnn_dig_count += sum(c.isdigit() for c in entry.generated_sum)
            tr_dig_count += sum(c.isdigit() for c in entry.text_rank_sum)
        print("CNN dig count= ", cnn_dig_count)
        print("TextRank dig count= ", tr_dig_count)

    def main(self):
        # Runs all preprocessing and evaluations to collect results

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

        # Prints three example summaries
        for x in range(3):
            print("CNN summary: ", self.pre.test_entries[x].generated_sum)
            print("TextRank summary: ", self.pre.test_entries[x].text_rank_sum)
            print("Combined summary: ", self.pre.test_entries[x].combined_sum)
            print("Gold Standard summary: ", self.pre.test_entries[x].summary)
            print("Doc ID: ",self.pre.test_entries[x].doc_id)
            print()

        plotter = Plotting(self.pre.test_entries)
        plotter.plot()

    def cnn(self, train):
        # Trains or loads CNN depending on train = True/False

        if train:
            print("getting initial salience scores...")
            self.pre.get_salience_scores(load=True)

            train_data, train_labels, _ = self.pre.get_cnn_vectors(train)

            assert(train_data is not None and train_labels is not None)

            # We've now pre process the documents, so now we can feed into the CNN

            TextCNN(train_data=train_data, train_labels=train_labels, num_filters=400, kernel_size=3)
            exit()
        else:
            _, _, test_data = self.pre.get_cnn_vectors(train)

            print("generating salience results...")

            gen_salience_results = TextCNN.eval(test_data)
            print(gen_salience_results)

            print("post-processing results...")
            self.post.get_results(gen_salience_results)

            print("getting summaries...")

            self.post.get_summary_sentences(100)
            self.post.return_summaries()


if __name__ == '__main__':
    Main().main()
