import numpy as np
from collections import defaultdict


class Perceptron_POS_Tagger(object):
    def __init__(self,):
        """ Initialize what we need to move forward (weights, vocabulary, tag-set and feature set)
        """
        self.tag_set = ['WRB', 'VBP', 'RBS', 'NNS', 'RB', '$', ':', '-LRB-', 'NN', 'IN', 'VBG', 'WDT', 'VBZ', 'UH',
                        'NNPS', ',', 'PDT', 'POS', 'RP', 'FW', 'JJ', 'WP$', 'LS', 'TO', 'RBR', '#', 'NNP', 'WP',
                        '-RRB-', '.', 'SYM', 'VBD', 'PRP', 'JJR', 'CC', 'VB', 'DT', 'MD', 'JJS', "''", '``', 'PRP$',
                        'EX', 'CD', 'VBN']
        self.tag_dict = {}
        self.pos_count = len(self.tag_set)
        self.weights = defaultdict()
        self.m = defaultdict()
        self.vocab = set({})
        self.feature_set = set({})

    def viterbi(self, sent):
        """ Implementation of the Viterbi algorithm using NumPy
        """
        transition = np.zeros([self.pos_count, self.pos_count])
        for p in self.tag_dict:
            given = self.tag_dict[p]
            prev_tag = "prev_tag=" + given
            transition[p] = self.weights[prev_tag]

        # initialization step
        v = np.zeros([self.pos_count, len(sent.snt)])
        backpointer = np.zeros([self.pos_count, len(sent.snt)])
        step = 0
        mtrx = [self.weights[ft] for ft in sent.ftlist[step]]
        init_vec = np.sum(mtrx, axis=0)
        v[:, step] = init_vec

        # intermediate steps
        for i in sent.ftlist[1:]:
            step += 1
            mtrx = [self.weights[ft] for ft in i]
            next_vec = np.sum(mtrx, axis=0)
            v[:, step] = np.amax(v[:, step-1, None] + next_vec + transition, axis=0)
            backpointer[:, step] = np.argmax(v[:, step-1, None] + next_vec + transition, axis=0)

        # termination step
        best_path_point = np.argmax(v[:, step])
        best_path = [best_path_point]

        while step > 0:
            previous = backpointer[int(best_path_point), step]
            best_path.append(int(previous))
            best_path_point = previous
            step -= 1

        return [self.tag_dict[i] for i in reversed(best_path)]

    def tag(self, test_data):
        """ Calls the Viterbi decoding algorithm to tag test_data.
        """
        tagged = []
        for sent in test_data:
            tagged_sent = []
            pred_tags = self.viterbi(sent)
            for word, tag in zip(sent.text, pred_tags):
                tagged_sent.append(word + '_' + tag)
            tagged.append(tagged_sent)
        return tagged

    def train(self, train_data, dev_data):
        """ Implementation of the Perceptron training algorithm.
        """
        self.init_weights()
        inverted_pos = {v: k for k, v in self.tag_dict.items()}

        # Averaged perceptron algorithm, hard-coded to run 5 epochs per experimental results
        for i in range(5):
            for training_example in train_data:
                tags = self.viterbi(training_example)
                real_tags = [t[1] for t in training_example.snt]
                pair_num = 0
                ll = list(zip(real_tags, tags))

                for real, guess in ll:
                    if real != guess:
                        real_ind = inverted_pos[real]
                        guess_ind = inverted_pos[guess]
                        for ft in training_example.ftlist[pair_num]:
                            self.weights[ft][real_ind] += 1
                            self.weights[ft][guess_ind] -= 1
                        if pair_num > 0:
                            prev_guess = ll[pair_num-1][1]
                            prev_f = 'prev_tag=' + prev_guess
                            self.weights[prev_f][real_ind] += 1
                            self.weights[prev_f][guess_ind] -= 1
                    pair_num += 1
            for w in self.weights:
                self.m[w] += self.weights[w]
            print('epoch ' + str(1 + i) + ' complete')

        for w in self.m:
            self.m[w] /= 5
        self.weights = self.m

    def collect_data(self, train_data):
        for sen in train_data:
            words = [w[0].lower() for w in sen.snt]
            self.vocab.update(words)
            for w in sen.ftlist:
                for ft in w:
                    self.feature_set.add(ft)
                    self.vocab.add(ft)

        tag_ind = 0
        for tag in self.tag_set:
            self.tag_dict[tag_ind] = tag
            tag_ind += 1

    def init_weights(self):
        for ft in self.feature_set:
            self.weights[ft] = np.zeros(len(self.tag_set))
            self.m[ft] = np.zeros(len(self.tag_set))
        for tag in self.tag_set:
            prev_feat = 'prev_tag=' + tag
            self.weights[prev_feat] = np.zeros(len(self.tag_set))
            self.m[prev_feat] = np.zeros(len(self.tag_set))

        self.weights['curr=UNK'] = np.zeros(len(self.tag_set))
        self.weights['prev_word1=UNK'] = np.zeros(len(self.tag_set))
        self.weights['prev_word2=UNK'] = np.zeros(len(self.tag_set))
        self.weights['next_word1=UNK'] = np.zeros(len(self.tag_set))
        self.weights['next_word2=UNK'] = np.zeros(len(self.tag_set))

        self.m['curr=UNK'] = np.zeros(len(self.tag_set))
        self.m['prev_word1=UNK'] = np.zeros(len(self.tag_set))
        self.m['prev_word2=UNK'] = np.zeros(len(self.tag_set))
        self.m['next_word1=UNK'] = np.zeros(len(self.tag_set))
        self.m['next_word2=UNK'] = np.zeros(len(self.tag_set))
