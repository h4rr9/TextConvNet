import pandas as pd
import numpy as np
import time
import re
import operator
import pickle as pk

from tqdm import tqdm
from gensim.models import KeyedVectors

tqdm.pandas()


def file2DataFrame(path):
    "saves subjectivity dataset into a csv for easy processing"

    sub_file = open(path + '\\quote.tok.gt9.5000')
    obj_file = open(path + '\\plot.tok.gt9.5000')

    sub_sentences = [line.strip() for line in sub_file]
    obj_sentences = [line.strip() for line in obj_file]

    sentences = sub_sentences + obj_sentences

    no_class_samples = len(sentences) // 2

    subjectivity = np.zeros(shape=(2 * no_class_samples), dtype=np.float32)
    subjectivity[:no_class_samples] = 1

    data = pd.DataFrame({'text': sentences, 'target': subjectivity},
                        columns=['text', 'target'])

    return data


def build_vocab(sentences, verbose=True):
    """
    :param sentences: list of list of words
    :return dictionary of words anf their counts
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
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()


print('Loading word2vec model')
start = time.time()

embeddings_index = KeyedVectors.load_word2vec_format(
    '.\\data\\word2vec\\GoogleNews-vectors-negative300.bin', binary=True)

print('Loaded word2vec model in %f seconds' % (time.time() - start))

train = file2DataFrame('.\\data\\subjectivity_dataset')

# no processing
sentences = train['text'].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)

print('no processing')
oov = check_coverage(vocab, embeddings_index)

print('preprocessing')

train['text'] = train['text'].progress_apply(lambda x: clean_text(x))
sentences = train['text'].apply(lambda x: x.split())

vocab = build_vocab(sentences)

print(len(vocab))
oov = check_coverage(vocab, embeddings_index)

# creating all required pickle and csv files
print('reducing embedding matrix to required vocabulary')

valid_words_dim = len(vocab) + 1  # for <EOF>

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
        embeddings_matrix[valid_word_index] = np.random.uniform(-0.25,0.25,300)
        valid_words.append(word)
        valid_word_index_map[word] = valid_word_index
        valid_word_index = valid_word_index + 1

valid_words.append('<EOF>')
embeddings_matrix[valid_word_index] = np.zeros(shape=(300, ), dtype=np.float32)
valid_word_index_map['<EOF>'] = valid_word_index

with open('.\\data\processed_data\\embeddings300_subjectivity.pkl', 'wb') as f:
    pk.dump(embeddings_matrix, f)

with open('.\\data\processed_data\\word2index_subjectivity.pkl', 'wb') as f:
    pk.dump(valid_word_index_map, f)

_sentences = [' '.join(sentence) for sentence in sentences]

train_df = pd.DataFrame(
    {'text': _sentences, 'target': train['target']}, columns=['text', 'target'])

train_df.to_csv('.\\data\\processed_data\\subjectivity.csv', index=False)
