class Sentence(object):
    def __init__(self, snt, v=None):
        """ Modify if necessary.
        """
        self.snt = snt
        self.training = False
        if v:
            self.vocab = v
            self.training = False
        else:
            self.training = True
        if type(self.snt[0]) is str:
            self.text = [s for s in self.snt]
        else:
            self.text = [w[0] for w in self.snt]
        self.ftlist = []
        for pos, word in enumerate(self.snt):
            fts = self.features(self.snt, pos)
            self.ftlist.append(fts)

    def features(self, sent, position):
        """ Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        """
        if type(sent[0]) is str:
            fts = []
            if self.training:
                curr_word = 'curr=' + sent[position].lower()
                fts.append(curr_word)
            elif sent[position].lower() in self.vocab:
                curr_word = 'curr=' + sent[position].lower()
                fts.append(curr_word)
            else:
                curr_word = 'curr=UNK'
                fts.append(curr_word)
            prefix = 'pref=' + sent[position][:2].lower()
            suffix = 'suff=' + sent[position][-2:].lower()
            if position == 0:
                prev_word1 = 'prev_word1=START'
                fts.append(prev_word1)
            if position >= 1:
                if self.training:
                    prev_word1 = 'prev_word1=' + sent[position - 1].lower()
                    fts.append(prev_word1)
                elif 'prev_word1=' + sent[position - 1].lower() in self.vocab:
                    prev_word1 = 'prev_word1=' + sent[position - 1].lower()
                    fts.append(prev_word1)
                else:
                    prev_word1 = 'prev_word1=UNK'
                    fts.append(prev_word1)

            if position >= 2:
                if self.training:
                    prev_word2 = 'prev_word2=' + sent[position - 2].lower()
                    fts.append(prev_word2)
                elif 'prev_word2=' + sent[position - 2].lower() in self.vocab:
                    prev_word2 = 'prev_word2=' + sent[position - 2].lower()
                    fts.append(prev_word2)
                else:
                    prev_word2 = 'prev_word2=UNK'
                    fts.append(prev_word2)

            if position <= (len(sent) - 2):
                if self.training:
                    next_word1 = 'next_word1=' + sent[position + 1].lower()
                    fts.append(next_word1)
                elif 'next_word1=' + sent[position + 1].lower() in self.vocab:
                    next_word1 = 'next_word1=' + sent[position + 1].lower()
                    fts.append(next_word1)
                else:
                    next_word1 = 'next_word1=UNK'
                    fts.append(next_word1)
            if position <= (len(sent) - 3):
                if self.training:
                    next_word2 = 'next_word2=' + sent[position + 2].lower()
                    fts.append(next_word2)
                elif 'next_word2=' + sent[position + 2].lower() in self.vocab:
                    next_word2 = 'next_word2=' + sent[position + 2].lower()
                    fts.append(next_word2)
                else:
                    next_word2 = 'next_word2=UNK'
                    fts.append(next_word2)

            if self.training:
                fts.append(prefix)
            elif prefix in self.vocab:
                fts.append(prefix)
            if self.training:
                fts.append(suffix)
            elif suffix in self.vocab:
                fts.append(suffix)

        else:
            fts = []
            if self.training:
                curr_word = 'curr=' + sent[position][0].lower()
                fts.append(curr_word)
            elif sent[position][0].lower() in self.vocab:
                curr_word = 'curr=' + sent[position][0].lower()
                fts.append(curr_word)
            else:
                curr_word = 'curr=UNK'
                fts.append(curr_word)
            prefix = 'pref=' + sent[position][0][:2].lower()
            suffix = 'suff=' + sent[position][0][-2:].lower()
            if position == 0:
                prev_word1 = 'prev_word1=START'
                fts.append(prev_word1)

            if position >= 1:
                if self.training:
                    prev_word1 = 'prev_word1=' + sent[position-1][0].lower()
                    fts.append(prev_word1)
                elif 'prev_word1=' + sent[position-1][0].lower() in self.vocab:
                    prev_word1 = 'prev_word1=' + sent[position-1][0].lower()
                    fts.append(prev_word1)
                else:
                    prev_word1 = 'prev_word1=UNK'
                    fts.append(prev_word1)

            if position >= 2:
                if self.training:
                    prev_word2 = 'prev_word2=' + sent[position-2][0].lower()
                    fts.append(prev_word2)
                elif 'prev_word2=' + sent[position-2][0].lower() in self.vocab:
                    prev_word2 = 'prev_word2=' + sent[position-2][0].lower()
                    fts.append(prev_word2)
                else:
                    prev_word2 = 'prev_word2=UNK'
                    fts.append(prev_word2)

            if position <= (len(sent) - 2):
                if self.training:
                    next_word1 = 'next_word1=' + sent[position+1][0].lower()
                    fts.append(next_word1)
                elif 'next_word1=' + sent[position+1][0].lower() in self.vocab:
                    next_word1 = 'next_word1=' + sent[position+1][0].lower()
                    fts.append(next_word1)
                else:
                    next_word1 = 'next_word1=UNK'
                    fts.append(next_word1)
            if position <= (len(sent) - 3):
                if self.training:
                    next_word2 = 'next_word2=' + sent[position+2][0].lower()
                    fts.append(next_word2)
                elif 'next_word2=' + sent[position+2][0].lower() in self.vocab:
                    next_word2 = 'next_word2=' + sent[position + 2][0].lower()
                    fts.append(next_word2)
                else:
                    next_word2 = 'next_word2=UNK'
                    fts.append(next_word2)

            if self.training:
                fts.append(prefix)
            elif prefix in self.vocab:
                fts.append(prefix)
            if self.training:
                fts.append(suffix)
            elif suffix in self.vocab:
                fts.append(suffix)

        return fts
