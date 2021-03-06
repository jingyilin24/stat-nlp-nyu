import heapq
from itertools import izip
import numpy as np

def ugly_normalize(vecs):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers==0]=1
    return (vecs.T / normalizers).T

class Embeddings:
    def __init__(self, vecsfile, vocabfile=None, normalize=True):
        if vocabfile is None: vocabfile = vecsfile.replace("npy","vocab")
        self._vecs = np.load(vecsfile)
        self._vocab = file(vocabfile).read().split()
        self.vocab = self._vocab
        if normalize:
            self._vecs = ugly_normalize(self._vecs)
            self._w2v = {w:i for i,w in enumerate(self._vocab)}

    @classmethod
    def load(cls, vecsfile, vocabfile=None):
        return Embeddings(vecsfile, vocabfile)

    def word2vec(self, w):
        return self._vecs[self._w2v[w]]

    def similar_to_vec(self, v, N=10):
        sims = self._vecs.dot(v)
        sims = heapq.nlargest(N, zip(sims,self._vocab,self._vecs))
        return sims
    
    def similarity(self, w1, w2):
        w1_vec = self.word2vec(w1)
        w2_vec = self.word2vec(w2)
        return np.dot(w1_vec, w2_vec) / (np.linalg.norm(w1_vec) * np.linalg.norm(w2_vec))
    
    def most_similar(self, word, N=10):
        w = self._vocab.index(word)
        sims = self._vecs.dot(self._vecs[w])
        sims = heapq.nlargest(N, zip(sims,self._vocab))
        return sims

    def analogy(self, pos1, neg1, pos2,N=10,mult=True):
        wvecs, vocab = self._vecs, self._vocab
        p1 = vocab.index(pos1)
        p2 = vocab.index(pos2)
        n1 = vocab.index(neg1)
        if mult:
            p1,p2,n1 = [(1+wvecs.dot(wvecs[i]))/2 for i in (p1,p2,n1)]
            if N == 1:
                return max(((v,w) for v,w in izip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
            return heapq.nlargest(N,((v,w) for v,w in izip((p1 * p2 / n1),vocab) if w not in [pos1,pos2,neg1]))
        else:
            p1,p2,n1 = [(wvecs.dot(wvecs[i])) for i in (p1,p2,n1)]
            if N == 1:
                return max(((v,w) for v,w in izip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))
            return heapq.nlargest(N,((v,w) for v,w in izip((p1 + p2 - n1),vocab) if w not in [pos1,pos2,neg1]))

