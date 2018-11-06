# -*- coding=utf8 -*-
import numpy as np
import json
import pickle


def load_data(path='math_90k.npz', test_split=0.1, seed=113, num_words=None,
              num_sfs=None, start_word=1, oov_word=2, word_index_from=3,
              sf_len=10, start_sf=1, end_sf=2, sf_index_from=1):
    np.random.seed(seed)

    with np.load(path) as f:
        brs = f['brs']
        ms = f['ms']
        sfs = f['sfs']

    # brs, sfs, index = share(brs, sfs)
    # num_sfs += index

    indices = np.arange(len(brs))
    np.random.shuffle(indices)
  
    brs = brs[indices]
    ms = ms[indices]
    sfs = sfs[indices]

    input_dic = open('shared.txt', 'r')
    dic = eval(input_dic.readline())
    input_dic.close()

    common_words = {}
    original_words = []
    original_tags = []
    unique_words = []
    unique_tags = []
    word_map_dic = {}
    tag_map_dic = {}

    if not num_words:
        num_words = max([max(x) for x in brs])

    if not num_sfs:
        num_sfs = max([max(x) for x in sfs])

  
    brs = [[w if(w < num_words) else 0 for w in x] for x in brs]
    sfs = [filter(lambda x: x < num_sfs, sf) for sf in sfs]

    unk_mask = [[0 if (w == 0) else 1 for w in x] for x in brs]
    brs_lens = [len(br) for br in brs]
    print(min(brs_lens), max(brs_lens), sum(brs_lens) * 1.0 / len(brs_lens))
    # START mask: 0, UNK mask: 0
    ms = [[0] + m for m in ms]
    ms = [[a * b for a, b in zip(al, bl)] for al, bl in zip(ms, unk_mask)]

    sf_lens = [len(sf) for sf in sfs]
    print(min(sf_lens), max(sf_lens), sum(sf_lens) * 1.0 / len(sf_lens))

    for x in brs:
        for w in x:
            if w == 0:
                continue
            original_words.append(w)

    for x in sfs:
        for t in x:
            original_tags.append(t)

    original_words = list(set(original_words))
    original_tags = list(set(original_tags))

    keys = dic.keys()
    for w in original_words:
        if w in keys:
            t = dic[w]
            if t in original_tags:
                common_words[w] = t
                continue
        unique_words.append(w)
    values = common_words.values()
    for t in original_tags:
        if t not in values:
            unique_tags.append(t)

    index = 3
    for uw in unique_words:
        word_map_dic[uw] = index
        index += 1
 
    tag_from = index
    shared_index = index
    for w in common_words.keys():
        word_map_dic[w] = index
        index += 1

    for w in common_words.values():
        tag_map_dic[w] = shared_index
        shared_index += 1

    for ut in unique_tags:
        tag_map_dic[ut] = shared_index
        shared_index += 1

    sfs_end = shared_index
    sfs_start = shared_index + 1

  
    new_brs = np.array([[start_word] + [word_map_dic.get(w, oov_word) for w in x]for x in brs])
    new_sfs = np.array([[tag_map_dic[t] for t in x]for x in sfs])

    split_index = int(len(brs) * test_split)

   
    new_sfs = [sf[:sf_len - 1] for sf in new_sfs]
 
    sfs_in = [[start_sf] + sf for sf in new_sfs]
    sfs_out = [sf + [end_sf] for sf in new_sfs]

    shared_map = open('shared_map.txt', 'w')
    shared_map.write(str(word_map_dic) + '\n')
    shared_map.write(str(tag_map_dic))
    print('unique_words:', len(unique_words))
    print('common:', len(common_words))
    print('unique_tags:', len(unique_tags))
    print('index: ', index)
    print('tags index:', index - len(common_words))
    print('shared_index:', shared_index)
    print('tag from :', tag_from)
    print('sfs end:', sfs_end)
    print('sfs start:', sfs_start)

    return (new_brs[:-split_index], ms[:-split_index], sfs_in[:-split_index], sfs_out[:-split_index]), \
           (new_brs[-split_index:], ms[-split_index:], sfs_in[-split_index:], sfs_out[-split_index:])


def load_topic(length, path):
    tmp_topic = []
    topics = pickle.load(open(path, "rb"))
    for index in range(length):
        tmp_topic.append(topics[index])
    topic = np.array(tmp_topic)
    return topic

def get_word_index(path='aspectj_word_index.json'):
    f = open(path)
    data = {}
    for l in f:
        d = l.split(' ')
        data[d[1]] = int(d[0])
    f.close()
    return data


def get_source_file_index(path='aspectj_source_file_index.json'):
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def count(path):
    with np.load(path) as f:
        sfs = f['sfs']
    sfs = sfs[:100000]
    tag_count = {}
    for sf in sfs:
        num = len(sf)
        if num in tag_count.keys():
            tag_count[num] += 1
        else:
            tag_count[num] = 1
    for c in tag_count:
        print(c, tag_count[c])


def test():
    (en_train, ms_train, de_train, y_train), (en_test, ms_test, de_test, y_test) = \
            load_data(path='SO350K.npz', num_words=10000, num_sfs=1003, start_sf=10290,
                      end_sf=10289, sf_len=6)

    print(en_train[0])
    print(ms_train[0])
    print(de_train[0])
    print(y_train[0])


if __name__ == "__main__":
    # load_data(path='SO350K.npz', num_words=10000, num_sfs=1003)
    count('SO350K.npz')

