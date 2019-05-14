import nltk
import numpy as np
import logging
from gensim.models.word2vec import Word2Vec
import re
from pyrouge import Rouge155
import os


class TrainWord2vec(object):

    @staticmethod
    def train_word2_vec(documents):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        doc_list = []
        for doc in documents:
            tokens = nltk.word_tokenize(doc)
            doc_list.append(tokens)

        model = Word2Vec(doc_list, size=100, window=5, min_count=0, workers=4)
        model.save('./word2vec_model.d2v')
        return model


class Postprocess(object):

    """
    This Class obtains summaries from output results obtained from
    each summarisation system
    """

    def __init__(self, pre):
        self.pre = pre

    def get_results(self, salience_scores):
        # Collects and sorts sentence results by score

        salience_scores = list(salience_scores)
        entries = self.pre.test_entries

        for e in entries:
            results = []

            for sent in e.sentences:
                results.append((sent, salience_scores.pop(0)[0]))

            assert(results is not [])

            e.output = sorted(results, key=lambda y: y[1], reverse=True)

    def calculate_rouge(self, alpha):
        # Calculates ROUGE-1 and ROUGE-2 scores between all generated
        # summaries and the gold standard summaries

        r = Rouge155("ROUGE-1.5.5", rouge_args="-e ROUGE-1.5.5/data -a -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0")

        r.system_dir = 'system_summaries'
        r.model_dir = 'model_summaries'
        r.system_filename_pattern = 'system_sum.(\d+).txt'
        r.model_filename_pattern = 'model_sum.[A-Z].#ID#.txt'

        # Writing model summaries
        list(map(os.unlink, (os.path.join("model_summaries", f) for f in os.listdir("model_summaries"))))
        for x in range(len(self.pre.test_entries)):
            entry = self.pre.test_entries[x]
            model_sum = entry.summary
            sentences = nltk.sent_tokenize(model_sum)
            file = open("model_summaries/model_sum.A." + str(x) + ".txt", "w+")
            for s in sentences:
                file.write(s + "\n")
            file.close()

        # Calculating ROUGE for CNN
        list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))

        for x in range(len(self.pre.test_entries)):
            entry = self.pre.test_entries[x]

            sentences = nltk.sent_tokenize(entry.generated_sum)
            file = open("system_summaries/system_sum." + str(x) + ".txt", "w+")
            for s in sentences:
                file.write(s + "\n")
            file.close()

        results_cnn = r.convert_and_evaluate()

        # Calculating ROUGE for TextRank
        list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))
        r = Rouge155("ROUGE-1.5.5", rouge_args="-e ROUGE-1.5.5/data -a -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0")

        r.system_dir = 'system_summaries'
        r.model_dir = 'model_summaries'
        r.system_filename_pattern = 'system_sum.(\d+).txt'
        r.model_filename_pattern = 'model_sum.[A-Z].#ID#.txt'

        for x in range(len(self.pre.test_entries)):
            entry = self.pre.test_entries[x]

            sentences = nltk.sent_tokenize(entry.text_rank_sum)
            file = open("system_summaries/system_sum." + str(x) + ".txt", "w+")
            for s in sentences:
                file.write(s + "\n")
            file.close()

        results_tr = r.convert_and_evaluate()

        # Calculating ROUGE for combined extension
        list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))
        r = Rouge155("ROUGE-1.5.5", rouge_args="-e ROUGE-1.5.5/data -a -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0")

        r.system_dir = 'system_summaries'
        r.model_dir = 'model_summaries'
        r.system_filename_pattern = 'system_sum.(\d+).txt'
        r.model_filename_pattern = 'model_sum.[A-Z].#ID#.txt'

        for x in range(len(self.pre.test_entries)):
            entry = self.pre.test_entries[x]

            sentences = nltk.sent_tokenize(entry.combined_sum)
            file = open("system_summaries/system_sum." + str(x) + ".txt", "w+")
            for s in sentences:
                file.write(s + "\n")
            file.close()

        results_combined = r.convert_and_evaluate()

        # Calculating ROUGE for control case
        list(map(os.unlink, (os.path.join("system_summaries", f) for f in os.listdir("system_summaries"))))
        r = Rouge155("ROUGE-1.5.5", rouge_args="-e ROUGE-1.5.5/data -a -n 2 -u -c 95 -x -r 1000 -f A -p 0.5 -t 0")

        r.system_dir = 'system_summaries'
        r.model_dir = 'model_summaries'
        r.system_filename_pattern = 'system_sum.(\d+).txt'
        r.model_filename_pattern = 'model_sum.[A-Z].#ID#.txt'

        for x in range(len(self.pre.test_entries)):
            entry = self.pre.test_entries[x]

            sentences = nltk.sent_tokenize(entry.control_sum)
            file = open("system_summaries/system_sum." + str(x) + ".txt", "w+")
            for s in sentences:
                file.write(s + "\n")
            file.close()

        results_control = r.convert_and_evaluate()

        print("\nCNN Scores:\n", results_cnn, "\n")
        print("TextRank Scores:\n", results_tr, "\n")
        print("Combined Scores:\n", results_combined, "\n")
        print("Control case scores\n", results_control)

    def get_summary_sentences(self, word_count):
        # Collects first sentences filling the desired word count

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
        # Outputs summary to entry object

        for entry in self.pre.test_entries:
            summary = ""
            sents = [tup[0] for tup in entry.output]
            for sent in sents:
                if re.search('[a-zA-Z0-9]', sent) is not None:
                    index = None
                    for i in range(len(sent)):
                        if sent[i].isalnum():
                            index = i
                            break
                    new_sent = sent[index:]
                    summary += new_sent + " "

            assert(len(nltk.word_tokenize(summary)) >= 100)

            entry.generated_sum = summary
