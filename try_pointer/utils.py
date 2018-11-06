import numpy as np


def encode_one_hot(int_data, vocab_size):
    one_hots = []
    for value in int_data:
        # print('value:', value)
        d = np.zeros(vocab_size)
        if value > 1:
            d[value-2] = 1
        one_hots.append(d)

    return one_hots


def weight_one_hot(words, tags):
    one_hots = []
    for value in tags:
        d = []
        for w in words:
            if w == value:
                d.append(1)
            else:
                d.append(0)
        one_hots.append(d)

    return one_hots


def load_embeddings(fn):
    embeddings_index = {}
    f = open(fn)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %sword vectors.' % len(embeddings_index))
    
    return embeddings_index, len(embeddings_index.items()[0][1])


def init_embeddings_from(embeddings_source, word_index):
    embedding_dim = len(embeddings_source.items()[0][1])
    embedding_matrix = np.zeros((len(word_index), embedding_dim))
    # embedding_matrix = np.random.uniform(-0.5, 0.5, (len(word_index), embedding_dim))
    count = 0
    not_found_words = []
    for word, i in word_index.items():
        embedding_vector = embeddings_source.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            count += 1
        else:
            not_found_words.append(word)
    
    print("Found %s/%s words in embeddings source." % (count, len(word_index)))
    
    return embedding_matrix
