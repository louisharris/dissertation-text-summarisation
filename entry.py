class Entry(object):

    def __init__(self):
        self.dir = None
        self.doc_id = None
        self.doc = None
        self.sentences = None
        self.parsed_sentences = None
        self.vectors = None
        self.summary = None
        self.saliences = None
        self.output = None
        self.output_tr = None
        self.generated_sum = None
        self.text_rank_sum = None
        self.control_sum = None
        self.combined_sum = None
        self.rouge_scores_cnn = None
        self.rouge_scores_tr = None
        self.rouge_scores_control = None
