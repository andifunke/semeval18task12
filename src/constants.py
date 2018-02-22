""" defines constants and files often used in the project """
ID = '#id'
SET = 'set'
LABEL = 'correctLabelW0orW1'
WARRANT0 = 'warrant0'
WARRANT1 = 'warrant1'
REASON = 'reason'
CLAIM = 'claim'
DEBATE_TITLE = 'debateTitle'
DEBATE_INFO = 'debateInfo'

META = [ID, SET, LABEL]
CONTENT = [WARRANT0, WARRANT1, REASON, CLAIM, DEBATE_TITLE, DEBATE_INFO]
CONTENT_SWAP = [WARRANT1, WARRANT0, REASON, CLAIM, DEBATE_TITLE, DEBATE_INFO]
CONTENT_MIN = [WARRANT0, WARRANT1, REASON, CLAIM]
CONTENT_MIN_SWAP = [WARRANT1, WARRANT0, REASON, CLAIM]
KEYS = META + CONTENT

FILES = dict(
    dev='../data/dev/dev-full.txt',
    # dev_true='../data/dev/dev-only-labels.txt',
    test='../data/test/test-full.txt',
    # test_true='../data/test/test-only-labels.txt',
    train='../data/train/train-full.txt',
    train_swap='../data/train/train-w-swap-full.txt',
)
SUBSETS = ['train', 'dev', 'test']

EMB_DIR = '../embeddings/'
EMB_FILES = dict(
    w2v="embeddings_cache_file_word2vec.pkl.bz2",
    d2v="embeddings_cache_file_dict2vec.pkl.bz2",
    d2v_pf="embeddings_cache_file_dict2vec_prov_freq.pkl.bz2",
    d2v_pf_lc="embeddings_cache_file_dict2vec_prov_freq_lc.pkl.bz2",
    d2v_pf_lc2="embeddings_cache_file_dict2vec_prov_freq_lc2.pkl.bz2",
    ftx="embeddings_cache_file_fastText.pkl.bz2",
    ftx_pf="embeddings_cache_file_fastText_prov_freq.pkl.bz2",
    ftx_pf_lc="embeddings_cache_file_fastText_prov_freq_lc.pkl.bz2",
    ftx_pf_lc2="embeddings_cache_file_fastText_prov_freq_lc2.pkl.bz2",
    ce_cb_100="custom_embedding_cb_100.vec",
    ce_cb_100_lc="custom_embedding_cb_100_lc.vec",
    ce_cb_300="custom_embedding_cb_300.vec",
    ce_cb_300_lc="custom_embedding_cb_300_lc.vec",
    ce_sg_100="custom_embedding_sg_100.vec",
    ce_sg_100_lc="custom_embedding_sg_100_lc.vec",
    ce_sg_300="custom_embedding_sg_300.vec",
    ce_sg_300_lc="custom_embedding_sg_300_lc.vec",
    ce_cb_hs_100="custom_embedding_hs_cb_100.vec",
    ce_cb_hs_100_lc="custom_embedding_hs_cb_100_lc.vec",
    ce_cb_hs_300="custom_embedding_hs_cb_300.vec",
    ce_cb_hs_300_lc="custom_embedding_hs_cb_300_lc.vec",
    ce_sg_hs_100="custom_embedding_hs_sg_100.vec",
    ce_sg_hs_100_lc="custom_embedding_hs_sg_100_lc.vec",
    ce_sg_hs_300="custom_embedding_hs_sg_300.vec",
    ce_sg_hs_300_lc="custom_embedding_hs_sg_300_lc.vec",
    ce_cb_i20_100="custom_embedding_iter20_cb_100.vec",
    ce_cb_i20_100_lc="custom_embedding_iter20_cb_100_lc.vec",
    ce_cb_i20_300="custom_embedding_iter20_cb_300.vec",
    ce_cb_i20_300_lc="custom_embedding_iter20_cb_300_lc.vec",
    ce_sg_i20_100="custom_embedding_iter20_sg_100.vec",
    ce_sg_i20_100_lc="custom_embedding_iter20_sg_100_lc.vec",
    ce_sg_i20_300="custom_embedding_iter20_sg_300.vec",
    ce_sg_i20_300_lc="custom_embedding_iter20_sg_300_lc.vec",
    ce_cb_hs_i25_100="custom_embedding_hs_iter25_cb_100.vec",
    ce_cb_hs_i25_100_lc="custom_embedding_hs_iter25_cb_100_lc.vec",
    ce_cb_hs_i25_300="custom_embedding_hs_iter25_cb_300.vec",
    ce_cb_hs_i25_300_lc="custom_embedding_hs_iter25_cb_300_lc.vec",
    ce_sg_hs_i25_100="custom_embedding_hs_iter25_sg_100.vec",
    ce_sg_hs_i25_100_lc="custom_embedding_hs_iter25_sg_100_lc.vec",
    ce_sg_hs_i25_300="custom_embedding_hs_iter25_sg_300.vec",
    ce_sg_hs_i25_300_lc="custom_embedding_hs_iter25_sg_300_lc.vec",
    ce_ftx_cb_hs_i20_100="custom_embedding_ftx_hs_iter20_cb_100.vec",
    ce_ftx_cb_hs_i20_100_lc="custom_embedding_ftx_hs_iter20_cb_100_lc.vec",
    ce_ftx_cb_hs_i20_300="custom_embedding_ftx_hs_iter20_cb_300.vec",
    ce_ftx_cb_hs_i20_300_lc="custom_embedding_ftx_hs_iter20_cb_300_lc.vec",
    ce_ftx_sg_hs_i20_100="custom_embedding_ftx_hs_iter20_sg_100.vec",
    ce_ftx_sg_hs_i20_100_lc="custom_embedding_ftx_hs_iter20_sg_100_lc.vec",
    ce_ftx_sg_hs_i20_300="custom_embedding_ftx_hs_iter20_sg_300.vec",
    ce_ftx_sg_hs_i20_300_lc="custom_embedding_ftx_hs_iter20_sg_300_lc.vec",
    ce_wiki_lc="custom_embedding_w2v_hs_iter05_sg_300_lc_wiki.vec",
    ce_wiki_i20_100_lc="custom_embedding_w2v_hs_iter20_sg_100_lc_wiki.vec",
    ce_wiki_i20_300_lc="custom_embedding_w2v_hs_iter20_sg_300_lc_wiki.vec",
)


