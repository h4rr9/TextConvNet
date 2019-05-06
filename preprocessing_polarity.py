import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
import operator
import time
import re
import pickle as pk

tqdm.pandas()


def file2DataFrame(path):
    "saves the polarity dataset into a csv file for easy processing"
    pos_file = open(path + '\\' + 'rt-polarity.pos', 'r')
    neg_file = open(path + '\\' + 'rt-polarity.neg', 'r')

    pos_lines = [line.strip() for line in pos_file]
    neg_lines = [line.strip() for line in neg_file]

    sentences = pos_lines + neg_lines

    no_class_samples = len(sentences) // 2

    sentiment = np.zeros(shape=(2 * no_class_samples,), dtype=np.float32)

    sentiment[:no_class_samples] = 1

    data = pd.DataFrame({'text': sentences, 'target': sentiment},
                        columns=['text', 'target'])

    return data


def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of word
    :return dictionary of words and their counts
    """
    vocab = {}

    for sentence in tqdm(sentences, disable=(not verbose)):
        for word in sentence:
            try:
                vocab[word] = vocab[word] + 1
            except KeyError:
                vocab[word] = 1

    return vocab


def check_coverage(vocab, embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0

    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except KeyError:
            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x


def clean_text(string):
    string = re.sub("[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub("\'s", " \'s", string)
    string = re.sub("\'ve", " \'ve", string)
    string = re.sub("n\'t", " n\'t", string)
    string = re.sub("\'re", " \'re", string)
    string = re.sub("\'d", " \'d", string)
    string = re.sub("\'ll", " \'ll", string)
    string = re.sub(",", " , ", string)
    string = re.sub("!", " ! ", string)
    string = re.sub("\(", " \( ", string)
    string = re.sub("\)", " \) ", string)
    string = re.sub("\?", " \? ", string)
    string = re.sub("\s{2,}", " ", string)

    return string.strip().lower()

train = file2DataFrame('.\\data\\polarity_dataset')

sentences = train['text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)

print('Loading word2vec model')
start = time.time()

embeddings_index = KeyedVectors.load_word2vec_format(
    '.\\data\\word2vec\\GoogleNews-vectors-negative300.bin', binary=True)

print('Loaded word2vec model in %f seconds' % (time.time() - start))
print('no processing')
print(len(vocab))
oov = check_coverage(vocab, embeddings_index)

print('preprocessing')

train['text'] = train['text'].progress_apply(lambda x: clean_text(x))
sentences = train['text'].apply(lambda x: x.split())

vocab = build_vocab(sentences)

print(len(vocab))
oov = check_coverage(vocab, embeddings_index)

print('reducing embedding matrix to required vocabulary')

valid_words_dim = len(vocab) + 1  # for <UNK> <EOF>


oov_list = [pair[0] for pair in oov]

embeddings_matrix = np.zeros(shape=(valid_words_dim, 300), dtype=np.float32)

valid_word_index = 0
valid_words = []
valid_word_index_map = {}
for word in vocab.keys():
    if word not in oov_list:
        embeddings_matrix[valid_word_index] = embeddings_index[word]
        valid_words.append(word)
        valid_word_index_map[word] = valid_word_index
        valid_word_index = valid_word_index + 1
    else:
        embeddings_matrix[valid_word_index] = np.random.random(300)
        valid_words.append(word)
        valid_word_index_map[word] = valid_word_index
        valid_word_index = valid_word_index + 1

valid_words.append('<EOF>')
embeddings_matrix[valid_word_index] = np.zeros(shape=(300, ), dtype=np.float32)
valid_word_index_map['<EOF>'] = valid_word_index

with open('.\\data\processed_data\\embeddings300_polarity.pkl', 'wb') as f:
    pk.dump(embeddings_matrix, f)

with open('.\\data\processed_data\\word2index_polarity.pkl', 'wb') as f:
    pk.dump(valid_word_index_map, f)

_sentences = [' '.join(sentence) for sentence in sentences]

train_df = pd.DataFrame(
    {'text': _sentences, 'target': train['target']}, columns=['text', 'target'])

train_df.to_csv('.\\data\\processed_data\\polarity.csv', index=False)
