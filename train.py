import datetime
import os
import time

import tensorflow as tf
import numpy as np
from tensorflow.python.platform.flags import FLAGS

from text_cnn import TextCNN


class Train:

    def __init__(self, x_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
    def start_session(self):
        # Creating new graph and setting it as default
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Code that operates on the default graph and session comes here...

                cnn = TextCNN(
                    sequence_length=self.x_train.shape[1],
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

                def train_step(self, x_batch, y_batch):
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

                def dev_step(self, x_batch, y_batch, writer=None):
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
                    zip(self.x_train, y_train), FLAGS.batch_size, FLAGS.num_epochs)
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
