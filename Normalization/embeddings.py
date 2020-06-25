

from gensim.models import KeyedVectors
import numpy as np
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)




class MimicIIEmbeddingVectorizer(object):
    def __init__(self, word2vec= None):
        if word2vec is None:
            self.word2vec = KeyedVectors.load('models/mimic_model_non_alpha_size100')
        else:
            self.word2vec = word2vec
        self.word2weight = None
        self.dim = self.word2vec.vector_size

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = collections.defaultdict(lambda: max_idf, [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return sparse.csr_matrix(np.array([
                np.mean([self.word2vec.wv[w] * self.word2weight[w]
                         for w in words if w in self.word2vec.wv] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ]))


