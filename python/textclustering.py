from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--lsa",
              dest="n_components", type="int",
              help="Preprocess documents with latent semantic analysis.")
op.add_option("--n-features", type=int, default=10000,
              help="Maximum number of features (dimensions)"
                   " to extract from text.")
print(__doc__)
op.print_help()

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

f = open('workfile', 'r')
dataset = f.readlines()

print("Using a tdifd vectorizer")
t0 = time()
vectorizer = TfidfVectorizer(max_features=opts.n_features, max_df=0.5,
                             min_df=2) #stop_words='russian'
X = vectorizer.fit_transform(dataset)

'''
# Выделение главных компонент!!!!!!!!!!! можно про это написать в дипломе))
svd = TruncatedSVD(n_components=500, random_state=666)
svd.fit_transform(X)

vocabulary_from_pca = np.array(vectorizer.get_feature_names())
for i in range(15):
    print('Top-words for {} component'.format(i + 1), end=': ')
    for word in vocabulary_from_pca[svd.components_[i] >= np.sort(svd.components_[i])[-10]]:
        print(word, end=' ')
    print()
'''
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()

'''
if opts.n_components:
    print("Performing dimensionality reduction using LSA")
    t0 = time()
    # Vectorizer results are normalized, which makes KMeans behave as
    # spherical k-means for better results. Since LSA/SVD results are
    # not normalized, we have to redo the normalization.
    svd = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X = lsa.fit_transform(X)

    print("done in %fs" % (time() - t0))

    explained_variance = svd.explained_variance_ratio_.sum()
    print("Explained variance of the SVD step: {}%".format(
        int(explained_variance * 100)))

    print()
'''
true_k=50

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1,
            verbose=False, random_state = 70)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
#print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
#print("Adjusted Rand-Index: %.3f"
      #% metrics.adjusted_rand_score(labels, km.labels_))
#print("Silhouette Coefficient: %0.3f"
 #     % metrics.silhouette_score(X, km.labels_, sample_size=1000))

#print()

print("Top terms per cluster:")
'''
if opts.n_components:
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
else:
'''

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :15]:
        print(' %s' % terms[ind], end='')
    print()

y = km.predict(X)


# for cur in range(0, true_k):
#   print(len(np.where(y == cur)[0]), cur)

# # biggest cluster 9
# print(dataset)
# print(np.where(y == 9)[0])
# new_dataset = [dataset[index] for index in np.where(y == 9)[0]]
# XX = vectorizer.fit_transform(new_dataset)


# true_k=50

# km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1,
#             verbose=False, random_state = 70)

# print("Clustering sparse data with %s" % km)
# t0 = time()
# km.fit(XX)
# print("done in %0.3fs" % (time() - t0))

# print("Top terms per cluster:")

# order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i, end='')
#     for ind in order_centroids[i, :15]:
#         print(' %s' % terms[ind], end='')
#     print()

# y = km.predict(XX)



clusters_list = [
    [0,1,5,19,26,42 ],
    [2,4,7,8,10,15,17,18,20,21,23,25,32,34,35,40,48,49],
    [3,24 ],
    [6,12,33,46 ],
    [11,22,28,44 ],
    [13,16,29,41,43],
    [14,39 ],
    [27,37 ],
    [30,36 ],
    [45 ],
    [31 ],
    [38,47 ]
 ]
#to_delete_clusters=[9]
good_ones = [0,2,3,6,11,13,14,27,30,45,9,31,38]
for cur_cluster in clusters_list:
  sum_lens = 0
  for elem in cur_cluster:
    sum_lens += len(np.where(y == elem)[0])
  km.cluster_centers_[cur_cluster[0]] = km.cluster_centers_[cur_cluster[0]] * len(np.where(y == cur_cluster[0])[0]) / sum_lens
  for i in range(1, len(cur_cluster)):
    km.cluster_centers_[cur_cluster[0]] += km.cluster_centers_[cur_cluster[i]] * len(np.where(y == cur_cluster[i])[0]) / sum_lens
    km.cluster_centers_[cur_cluster[i]] = [-10 for i in range(len(km.cluster_centers_[cur_cluster[0]]))]

#for i in range(len(to_delete_clusters)):
   #km.cluster_centers_[to_delete_clusters[i]] = [-10 for i in range(len(km.cluster_centers_[0]))]

# #km.cluster_centers_[27] = (len0*km.cluster_centers_[27] + len1*km.cluster_centers_[35]) / (len0 + len1)
# #km.cluster_centers_[35] = [-10 for i in range(len(km.cluster_centers_[27]))]

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
f = open('clusters', 'w')
for i in good_ones:
    f.write("Cluster %d:" % i)
    for ind in order_centroids[i, :20]:
        f.write(' %s' % terms[ind])
    f.write('\n')

yy = km.predict(X)

my_map = {
    0: 0,
    2: 1,
    3: 2,
    6: 3,
    11: 4,
    13: 5,
    14: 6,
    27: 7,
    30: 8,
    45: 9,
    9: 10,
    31: 11,
    38: 12    
}

f = open('training_set', 'w')
for i in range(len(dataset)):
  f.write(dataset[i][:-1] + ' ' + str(my_map[yy[i]]) + '\n')

for cur in range(0, len(good_ones)):
  print(len(np.where(yy == good_ones[cur])[0]), good_ones[cur])


# 0 -- 0,1,5,19,26,42 — трансфер, авто, переезд
# 1 -- 2,4,7,8,10,15,17,18,20,21,23,25,32,34,35,40,48,49— ремонт, строительство
# 2 -- 3,24 — уборка,
# 3 -- 6,12,33,46 — бьюти
# 4 -- 11,22,28,44 — сад, огород
# 5 -- 13,16,29,41,43— бытовая, компьютерная техника
# 6 -- 14,39 — юриспруденция, адвокат
# 7 -- 27,37 — услуги репетиторства
# 8 -- 30,36 — торты, конфета-букет
# 9 -- 45 — фотограф свадебный
# 10 -- 9 — прочее(удалить?)
# 11 -- 31 — вывоз мусора
# 12 -- 38,47 — реставрация мебели, мебель перетяжка


# #for ind in np.argwhere(y != yy):
# #  print (dataset[ind[0]] + str(yy)
# for cur in good_ones:
#   print(len(np.where(yy == cur)[0]))
# print(yy)

# '''from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# clf_logreg = LogisticRegression()
# clf_logreg.fit(x_train, y_train)
# y_pred = clf_logreg.predict(x_test)
# '''
