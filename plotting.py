import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nltk
import numpy as np


class Plotting(object):

    def __init__(self, pre):
        self.pre = pre

    def plot(self):

        """
        Plots the ROUGE score to window size graph
        """
        x = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 94]
        y = [0.4343, 0.4412, 0.4356, 0.4008, 0.4161, 0.4178, 0.3986, 0.4017, 0.4105, 0.4179, 0.4233]
        y_2 = [.1748, .1814, .1771, .1410, 0.1562, 0.1593, 0.1396, 0.1426, 0.1499, 0.1553, 0.1615]

        f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

        # plot the same data on both axes
        ax.plot(x, y,)
        ax2.plot(x, y_2, color='green')
        z = np.polyfit(x, y, 2)
        f = np.poly1d(z)
        x_new = np.linspace(x[0], x[-1], 50)
        y_new = f(x_new)
        ax.plot(x_new, y_new,'r', linestyle='dashed', linewidth=1)

        z_2 = np.polyfit(x, y_2, 2)
        f_2 = np.poly1d(z_2)
        x2_new = np.linspace(x[0], x[-1], 50)
        y2_new = f_2(x_new)
        ax2.plot(x2_new, y2_new, 'r', linestyle='dashed', linewidth=1)

        blue_patch = mpatches.Patch(color='blue', label='ROUGE-1')
        green_patch = mpatches.Patch(color='green', label='ROUGE-2')
        ax.legend(handles=[blue_patch, green_patch])

        ax.set_ylim(.398, .45)  # ROUGE-1 data
        ax2.set_ylim(.14, .19)  # ROUGE-2 data

        ax.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax.xaxis.tick_top()
        ax.tick_params(labeltop='off')  # don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        ax2.set(xlabel='Window Size')
        ax2.set(ylabel='ROUGE Score', )
        ax2.yaxis.set_label_coords(-0.1, 1.0)

        d = .015

        # Plots graph split
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)
        ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ttl = ax.title
        ttl.set_position([.5, 1.05])

        plt.savefig("rouge.png")
        plt.show()

        # Plots Output score to sentence length graph
        sent_lengths = []
        sent_outs = []
        for entry in self.pre:
            ranking = entry.output
            for sent_score in ranking:
                sent_lengths.append(len(nltk.word_tokenize(sent_score[0])))
            sent_outs += [x[1] for x in ranking]

        x = sent_lengths
        y = sent_outs
        plt.scatter(x, y, s=1)
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'r')
        plt.xlabel("Sentence length")
        plt.ylabel("Output Score")
        # plt.savefig("sent_scores.png")
        plt.show()

        # Plots sentence length frequency density histogram
        sent_lengths = []
        for entry in self.pre:
            sents = entry.sentences
            for s in sents:
                sent_lengths.append(len(nltk.word_tokenize(s)))
        n, x, _ = plt.hist(sent_lengths, bins=20, density=True, histtype='step')
        bin_centers = 0.5 * (x[1:] + x[:-1])
        plt.plot(bin_centers, n)
        plt.xlabel("Sentence length")
        plt.ylabel("Frequency Density")
        # plt.savefig("sent_scores.png")
        plt.show()
