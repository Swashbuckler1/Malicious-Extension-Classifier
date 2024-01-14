from utils import *



from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN


with open('code1.txt', 'r') as f:
    code_section = ''.join(f.readlines())

tokenized_code = preprocess_and_tokenize(code_section)
print(len(tokenized_code))
word2vec_model = Word2Vec(sentences=tokenized_code, window=5, min_count=1, sg=0)
code_sections = [code_section]

section_embeddings = []
for section in code_sections:
    section_tokens = preprocess_and_tokenize(section)
    section_vector = [word2vec_model.wv[word] for word in section_tokens if word in word2vec_model.wv]
    if section_vector:
        section_embedding = np.mean(section_vector, axis=0)
        section_embeddings.append(section_embedding)

print(len(section_embeddings[0]))
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(section_embeddings)
print(labels)