import nltk
import numpy as np
from random import random, shuffle

from postprocessing import Postprocess
from preprocessing import Preprocessing
from text_cnn import TextCNN
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

        print("getting initial salience scores...")

        self.pre.get_salience_scores(0.5)

    def text_rank(self):
        test_entries = self.pre.test_entries
        textrank.TextRank.run(test_entries)

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

    def main(self):
        print("loading datasets...")
        self.load_dataset(stem=True)
        print("running control case model...")
        self.control_case()
        print("running TextRank model...")
        self.text_rank()
        print("running CNN model...")
        self.evaluate(train=False)

    def evaluate(self, train):

        train_data, train_labels, test_data = self.pre.get_cnn_vectors()
        # We've now pre process the documents, so now we can feed into the CNN

        print(len(test_data))

        if train:
            TextCNN(train_data=train_data, train_labels=train_labels, num_filters=400, kernel_size=3)
        else:

            print("generating salience results...")

            gen_salience_results = TextCNN.eval(test_data)
            print(gen_salience_results)

            print("post-processing results...")

            self.post.get_results(gen_salience_results)

            print("getting summaries...")

            self.post.get_summary_sentences(100)
            self.post.return_summaries()

            print("calculating rouge scores...")

            self.post.calculate_rouge(0.5)
            return
            mean_rouge_cnn = []
            for entry in self.pre.test_entries:
                mean_rouge_cnn.append(entry.rouge_scores_cnn)
            # print(mean_rouge_cnn)
            print("ROUGE results CNN = ", np.mean(mean_rouge_cnn, axis=0))

            mean_rouge_tr = []
            for entry in self.pre.test_entries:
                mean_rouge_tr.append(entry.rouge_scores_tr)
            print("ROUGE results TextRank = ", np.mean(mean_rouge_tr, axis=0))

            mean_rouge_control = []
            for entry in self.pre.test_entries:
                mean_rouge_control.append(entry.rouge_scores_control)
            print("ROUGE results Control = ", np.mean(mean_rouge_control, axis=0))


if __name__ == '__main__':
    Main().main()
