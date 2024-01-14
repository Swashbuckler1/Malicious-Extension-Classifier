from utils import *
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

malicious = r'data/extracted'
benign = r'data/extensions'

mal_data = read_data(malicious)
benign_data = read_data(benign)
print(len(mal_data), len(benign_data))
tokenized_code = preprocess_and_tokenize(mal_data)
tokenized_ben_code = preprocess_and_tokenize(benign_data)

word2vec_model = Word2Vec(sentences=tokenized_code, window=10, min_count=1, sg=0)
word2vec_model2 = Word2Vec(sentences=tokenized_ben_code, window=10, min_count=1, sg=0)
#print(tokenized_code[1])
section_embeddings = []
for section in tokenized_code:
    section_vector = [word2vec_model.wv[word] for word in section if word in word2vec_model.wv]
    if section_vector:
        section_embedding = np.mean(section_vector, axis=0)
        section_embeddings.append((section_embedding, 1))

for section in tokenized_ben_code:
    section_vector = [word2vec_model2.wv[word] for word in section if word in word2vec_model2.wv]
    if section_vector:
        section_embedding = np.mean(section_vector, axis=0)
        section_embeddings.append((section_embedding, 0))

section_embeddings_arr = np.array([(t[0], t[1]) for t in section_embeddings], dtype=object)
features = section_embeddings_arr[:, 0]
labels = section_embeddings_arr[:, 1]
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

features_flat = np.array([embedding.flatten() for embedding in features])
eps = 0.5
min_samples = 5
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_flat)

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(features_pca, labels)

ari = adjusted_rand_score(labels, cluster_labels)
print(f"Adjusted Rand Index: {ari}")

ami = adjusted_mutual_info_score(labels, cluster_labels)
print(f"Adjusted Mutual Info Score: {ami}")

unique_clusters, cluster_sizes = np.unique(cluster_labels, return_counts=True)
for cluster, size in zip(unique_clusters, cluster_sizes):
    print(f"Cluster {cluster}: {size} instances")

# plt.figure(figsize=(8, 6))
# palette = sns.color_palette("husl", n_colors=len(set(cluster_labels)))
#
# for cluster, color in zip(set(cluster_labels), palette):
#     cluster_points = [x for x, label in zip(range(len(labels)), cluster_labels) if label == cluster]
#     plt.scatter(cluster_points, [1] * len(cluster_points), color=color, label=f'Cluster {cluster}')
#
# true_points = [x for x, label in enumerate(labels) if label == 1]
# plt.scatter(true_points, [0.8] * len(true_points), color='gray', marker='x', label='Benign')
#
# true_points = [x for x, label in enumerate(labels) if label == 0]
# plt.scatter(true_points, [0.8] * len(true_points), color='black', marker='x', label='Malicious')
#
# plt.title('K-Means Clustering Evaluation')
# plt.xlabel('Data Points')
# plt.yticks([])
# plt.legend()
# plt.savefig('Clustering Viz.png')
# plt.show()

# print(section_embeddings[0])
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(features_pca)
# plt.scatter(features_pca[:, 0], features_pca[:, 1], c=labels_dbscan, cmap='viridis')
# plt.title('DBSCAN')
# plt.colorbar()
# plt.show()
#
# plt.figure(figsize=(8, 6))
# palette = sns.color_palette("husl", n_colors=len(set(labels_dbscan)))

# for cluster, color in zip(set(labels_dbscan), palette):
#     cluster_points = [x for x, label in zip(range(len(labels)), labels_dbscan) if label == cluster]
#     plt.scatter(cluster_points, [1] * len(cluster_points), color=color, label=f'Cluster {cluster}')
#
# true_points = [x for x, label in enumerate(labels) if label == 1]
# plt.scatter(true_points, [0.8] * len(true_points), color='black', marker='x', label='Malicious')
#
# true_points = [x for x, label in enumerate(labels) if label == 0]
# plt.scatter(true_points, [0.8] * len(true_points), color='gray', marker='x', label='Benign')

# plt.title('DBSCAN Clustering Evaluation')
# plt.xlabel('Data Points')
# plt.yticks([])
# plt.legend()
# plt.savefig('DBSCAN Viz.png')
# plt.show()

ari = adjusted_rand_score(labels, labels_dbscan)
print(f"Adjusted Rand Index: {ari}")

ami = adjusted_mutual_info_score(labels, labels_dbscan)
print(f"Adjusted Mutual Info Score: {ami}")

print(f'DBSCAN Accuracy: {np.mean(labels == labels_dbscan)}')
print(f'K-Means Accuracy: {np.mean(labels == cluster_labels)}')

x_train, x_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.20, random_state=42)
print(y_test)
svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)

y_pred = svm_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print(f"SVM Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with a blue color map
class_names = ['Benign', 'Malicious']
disp = ConfusionMatrixDisplay(cm,
                             display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title('SVM')
plt.savefig('svm_cm.png')
plt.show()

forest = RandomForestClassifier(n_estimators=100)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm,
                             display_labels=class_names)

disp.plot(cmap=plt.cm.Blues)
plt.title('Random Forest Classifier')
plt.savefig('rfc_cm.png')
plt.show()