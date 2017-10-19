import os
import numpy as np
import vocabulary_embeddings_extractor as vocex
import json

current_dir = os.getcwd()
# out_file = current_dir + "dict2vec_embedding_cache.json"
# embeddings_cache_file = current_dir + "/embeddings_cache_file_word2vec.pkl.bz2"
# freq, embedding_map = vocex.load_word_frequencies_and_embeddings(embeddings_cache_file)

d2v = r'/home/andreas/workspace/sem_train/remote_vectors/dict2vec/dict2vec-vectors-dim300.vec'
ftx = r'/home/andreas/workspace/sem_train/remote_vectors/fastText/wiki.en.vec'
vocex.prepare_word_embeddings_cache([current_dir + '/data/'], current_dir + '/embeddings_cache_file_dict2vec_prov_freq.pkl.bz2',
                                    embeddings_file_name=d2v)
vocex.prepare_word_embeddings_cache([current_dir + '/data/'], current_dir + '/embeddings_cache_file_fastText_prov_freq.pkl.bz2',
                                    embeddings_file_name=ftx)

# i = 0
# dic = dict()
# with open(path + file, 'r', encoding='utf-8') as f:
#     for line in f:
#         # print('line:', line)
#         token = line.split()
#         # print('token:', token)
#         word = token.pop(0)
#         # print('word:', word)
#         if word in embedding_map:
#             dic[word] = np.asfarray(token)
#             # print('dic:', dic)
#             i += 1
#             if i > 10:
#                 break
#
# with open(out_file, 'w') as of:
#     json.dump(dic, of)
