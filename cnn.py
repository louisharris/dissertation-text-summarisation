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

from tensorflow.python.platform.flags import FLAGS


def train_word2_vec(parameters, documents):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    docList = []
    for doc in documents:
        tokens = nltk.word_tokenize(doc)
        docList.append(tokens)

    model = Word2Vec(docList, size=100, window=5, min_count=0, workers=4)
    # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save('./word2vec_model.d2v')
    return model


# Creating map of each document with its sentences represented as vectors Returns MAP<STRING, [(SENT, VECTOR)]>
def create_word_vectors(model):
    zero_vector = np.zeros(300)
    total_sent_vecs = {}
    doc_sent_map = get_doc_sentence_map()  # MAP<DOCS, [SENTENCES]>
    doc_sent_map_copy = copy.deepcopy(doc_sent_map)
    for doc in doc_sent_map:
        for s in range(len(doc_sent_map[doc])):
            words = nltk.word_tokenize(doc_sent_map[doc][s])
            doc_sent_map[doc][s] = words

    doc_sent_map = pad_sentences(doc_sent_map)

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


# Creating map of each document with its sentences Returns MAP<DOCS, [SENTENCES]>
def get_doc_sentence_map():
    filepath = "duc2001_simplified/training/"
    doc_catalog = []
    for filename in os.listdir(filepath):
        if filename.endswith(".xml"):
            file = open('duc2001_simplified/training/' + filename)
            text = file.read()
            file.close()

            # Extracting text from the XML files
            soup = BeautifulSoup(text, 'html.parser')

            documents = soup.findAll('text')
            newSet = []
            for doc in range(len(documents)):
                newSet.append(str(documents[doc]).replace("<p>", " ").replace("</p>", " ").replace("<text>", " ") \
                              .replace("</text>", " ").replace("\n", " "))

            doc_catalog.append(newSet)

    train_reviews = {}
    for doc in doc_catalog:
        for text in doc:
            text_sentences = nltk.sent_tokenize(text)
            train_reviews[text] = text_sentences

    return train_reviews


# Calculating sentence salience score using Rouge metric. returns MAP<DOC,[(SENT, SCORE)]>
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


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Layers
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Scoring and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses)  #

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Creating new graph and setting it as default
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Code that operates on the default graph and session comes here...

                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=2,
                    vocab_size=len(vocabulary),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
                    num_filters=FLAGS.num_filters)

                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-4)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Summary stuff

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.scalar_summary("loss", cnn.loss)
                acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.merge_summary([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

                # Dev summaries
                dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

                # Checkpointing
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                # Tensorflow assumes this directory already exists so we need to create it
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.all_variables())

                # Initialising variables
                sess.run(tf.initialize_all_variables())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

                # Training loop
                # Generate batches
                batches = data_helpers.batch_iter(
                    zip(x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

def main():
    model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)

    doc_sentence_vectors = create_word_vectors(model)  # MAP<DOC, [(SENT, VECTOR)]> for doc, for array, for tuple

    salience_scores = get_salience_scores(get_doc_sentence_map(), 0.5)  # MAP<DOC,[(SENT, SCORE)]>  for doc, for tuple

    doc_sent_vec_salience_map = {}

    for doc in salience_scores:
        for tuple1 in doc_sentence_vectors[doc]:
            for tuple2 in salience_scores[doc]:
                if tuple1[0] == tuple2[0]:
                    doc_sent_vec_salience_map[doc] = (tuple1[1], tuple2[1])

    # We've now pre process the documents, so now we can feed into the CNN


if __name__ == '__main__':
    main()
