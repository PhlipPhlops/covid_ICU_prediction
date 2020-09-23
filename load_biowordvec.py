from gensim.models import KeyedVectors
print("Loading BioWordVec (it's 13G)...")
model = KeyedVectors.load_word2vec_format('./BioWordVec/BioWordVec_d200.vec.bin', binary=True)
