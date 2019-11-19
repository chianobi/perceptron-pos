import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence


def read_in_gold_data(filename, v=None):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line, v) for line in lines]

    return sents 


def read_in_plain_data(filename, v):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line, v) for line in lines]

    return sents 


def output_auto_data(auto_data, filename):
    """ According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...). 
    """
    with open(filename, 'w') as f:
        for sent in auto_data:
            lin = ''
            for tok in sent:
                lin += tok + ' '
            f.write(lin + '\n')


if __name__ == '__main__':

    # Run
    # python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt
    # to train & test your tagger
    train_file = sys.argv[1]
    gold_dev_file = sys.argv[2]
    plain_dev_file = sys.argv[3]
    test_file = sys.argv[4]

    # Read in data
    train_data = read_in_gold_data(train_file)
    my_tagger = Perceptron_POS_Tagger()
    my_tagger.collect_data(train_data)
    gold_dev_data = read_in_gold_data(gold_dev_file, my_tagger.vocab)
    plain_dev_data = read_in_plain_data(plain_dev_file, my_tagger.vocab)
    test_data = read_in_plain_data(test_file, my_tagger.vocab)

    # Train your tagger

    my_tagger.train(train_data, gold_dev_data)
    my_tagger.viterbi(train_data[3])

    # Apply your tagger on dev & test data
    auto_dev_data = my_tagger.tag(plain_dev_data)
    auto_test_data = my_tagger.tag(test_data)

    # Output your auto tagged data
    output_auto_data(auto_dev_data, 'dev.tagged')
    output_auto_data(auto_test_data, 'test.tagged')
