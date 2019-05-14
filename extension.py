import nltk
import textrank


class Extension(object):
    """
    This uses the algorithm mentioned in the report, to create
    a combined model selectively choosing sentences from each of
    TextRank and the CNN summaries.
    """

    def __init__(self, entries):
        for entry in entries:

            out_cnn = entry.output
            out_tr = entry.output_tr

            cnn_sum = textrank.TextRank.get_summary_sentences(150, out_cnn)
            tr_sum = textrank.TextRank.get_summary_sentences(150, out_tr)

            assert(cnn_sum is not None and tr_sum is not None)

            new_sum = ""
            cnn_sents = nltk.sent_tokenize(cnn_sum)
            tr_sents = nltk.sent_tokenize(tr_sum)

            sents_in_common = list(filter(lambda x: x in tr_sents, cnn_sents))
            tr_sents = list(filter(lambda x: x not in sents_in_common, tr_sents))

            used_sents = []
            word_count = 100
            for sent in sents_in_common:
                if word_count < 0:
                    break
                word_count -= len(nltk.word_tokenize(sent))
                used_sents.append(sent)

            while word_count > 0:
                sent = tr_sents.pop(0)
                if sent not in used_sents:
                    word_count -= len(nltk.word_tokenize(sent))
                    used_sents.append(sent)

            for sent in used_sents:
                new_sum += sent + " "

            assert(len(nltk.word_tokenize(new_sum)) >= 100)
            entry.combined_sum = new_sum

