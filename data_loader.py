import gzip
import os.path
# from typing import Dict, Tuple, List
import math

import vocabulary_embeddings_extractor

MAX_LEN = 0


def string_to_indices(string: str, word_to_indices_map_param: dict, nb_words: int = None, lc=False) -> list:
    """
    Tokenizes a string and converts to indices specified in word_to_indices_map; performs also OOV replacement
    :param string: string
    :param word_to_indices_map_param: map (word index, embedding index)
    :param nb_words: all words with higher index are treated as OOV
    :param lc: use lowercase on all tokens
    :return: a list of indices
    """
    tokens = vocabulary_embeddings_extractor.tokenize(string.lower() if lc else string)
    # global MAX_LEN
    # MAX_LEN = max(len(tokens), MAX_LEN)

    # now convert tokens to indices; set to 2 for OOV
    word_indices_list = [word_to_indices_map_param.get(word, 2) for word in tokens]

    # limit words to max nb_words (set them to OOV = 2):
    if nb_words:
        word_indices_list = [2 if word_index >= nb_words else word_index for word_index in word_indices_list]

    # print(string)
    #print(tokens)
    #print(word_indices_list)

    return word_indices_list


def load_single_instance_from_line(line: str, word_to_indices_map: dict, nb_words: int = None, lc=False,
                                   no_labels=False) -> tuple:
    """
    Load a single training/test instance from a single line int tab-separated format
    :param no_labels: set True if data does not provide labels
    :param lc: use lowercase on all tokens
    :param line: string line
    :param word_to_indices_map:  map (word index, embedding index)
    :param nb_words: all words with higher index are treated as OOV
    :return: tuple (instance_id, warrant0, warrant1, correct_label_w0_or_w1, reason, claim, debate_meta_data)
    """
    split_line = line.split('\t')
    # "#id warrant0 warrant1 correctLabelW0orW1 reason claim debateTitle debateInfo
    #print(line)
    #print(split_line)

    if no_labels:
        assert len(split_line) == 7
        i0, i1, i2, i3, i4, i5, i6, i7 = (0, 1, 2, -1, 3, 4, 5, 6)
        correct_label_w0_or_w1 = float('NaN')
    else:
        assert len(split_line) == 8
        i0, i1, i2, i3, i4, i5, i6, i7 = (0, 1, 2, 3, 4, 5, 6, 7)
        correct_label_w0_or_w1 = int(split_line[i3])

    instance_id = split_line[i0]
    warrant0 = string_to_indices(split_line[i1], word_to_indices_map, nb_words, lc)
    warrant1 = string_to_indices(split_line[i2], word_to_indices_map, nb_words, lc)
    reason = string_to_indices(split_line[i4], word_to_indices_map, nb_words, lc)
    claim = string_to_indices(split_line[i5], word_to_indices_map, nb_words, lc)
    debate_title = string_to_indices(split_line[i6], word_to_indices_map, nb_words, lc)
    debate_info = string_to_indices(split_line[i7], word_to_indices_map, nb_words, lc)
    # concatenate these two into one vector
    debate_meta_data = debate_title + debate_info

    # make sure everything is filled
    assert len(instance_id) > 0
    assert len(warrant0) > 0
    assert len(warrant1) > 0
    if no_labels:
        assert math.isnan(correct_label_w0_or_w1)
    else:
        assert correct_label_w0_or_w1 == 0 or correct_label_w0_or_w1 == 1
    assert len(reason) > 0
    assert len(claim) > 0
    assert len(debate_meta_data) > 0

    return instance_id, warrant0, warrant1, correct_label_w0_or_w1, reason, claim, debate_meta_data


def load_single_file(file_name: str, word_to_indices_map: dict, nb_words: int = None, lc=False,
                     no_labels=False) -> tuple:
    """
    Loads a single train/test file and returns a tuple of lists
    :param no_labels: set True if data does not provide labels
    :param lc: use lowercase on all tokens
    :param file_name: full file name
    :param word_to_indices_map: vocabulary map in form {word: index} where index also correspond to frequencies
    :param nb_words: if a particular word index is higher than this value, the word is treated as OOV
    :return: instance_id_list = list of strings
        warrant0_list = list of list of word indices (where integer is a word index taken from word_to_indices_map)
        warrant1_list = list of list of word indices
        correct_label_w0_or_w1_list = list of 0 or 1
        reason_list = list of list of word indices
        claim_list = list of list of word indices
        debate_meta_data_list = list of list of word indices
    """
    assert len(word_to_indices_map) > 0

    if file_name.endswith('gz'):
        f = gzip.open(file_name, 'rb')
    else:
        f = open(file_name, 'r')
    lines = f.readlines()
    # remove first line with comments
    del lines[0]

    instance_id_list = []
    warrant0_list = []
    warrant1_list = []
    correct_label_w0_or_w1_list = []
    reason_list = []
    claim_list = []
    debate_meta_data_list = []

    for line in lines:
        # convert to vectors of embedding indices where appropriate
        instance_id, warrant0, warrant1, correct_label_w0_or_w1, reason, claim, debate_meta_data = \
            load_single_instance_from_line(line, word_to_indices_map, nb_words, lc, no_labels)

        # add to the result
        instance_id_list.append(instance_id)
        warrant0_list.append(warrant0)
        warrant1_list.append(warrant1)
        correct_label_w0_or_w1_list.append(correct_label_w0_or_w1)
        reason_list.append(reason)
        claim_list.append(claim)
        debate_meta_data_list.append(debate_meta_data)

    # global MAX_LEN
    # print('MAX_LEN:', MAX_LEN)

    return (instance_id_list, warrant0_list, warrant1_list, correct_label_w0_or_w1_list, reason_list, claim_list,
            debate_meta_data_list)


def __main__():
    current_dir = os.getcwd()
    embeddings_cache_file = current_dir + "/embeddings_cache_file_word2vec.pkl.bz2"

    # load pre-extracted word-to-index maps and pre-filtered Glove embeddings
    word_to_indices_map, word_index_to_embeddings_map = \
        vocabulary_embeddings_extractor.load_cached_vocabulary_and_embeddings(embeddings_cache_file)

    load_single_file(current_dir + '/data/train.tsv', word_to_indices_map)


if __name__ == "__main__":
    __main__()

pass
