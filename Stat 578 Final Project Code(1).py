# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 17:50:18 2017

@author: Xinchen
"""
##############Libraries Used#################
import os
import pandas as pd
import numpy as np
import collections as cl
import seaborn as sns
import matplotlib.pyplot as plt
import string
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
from PIL import Image 
from pylab import rcParams
from stop_words import get_stop_words

##############Read in the dataset##############
os.chdir("C:/Users/Xinchen/Desktop/stat578 Qu")
dt = pd.read_csv("quora_duplicate_questions.tsv", sep='\t', index_col="id",\
                 usecols = [0,3,4,5])
##############Preprocessing##############
#Remove NA in the dataset
dt= dt.dropna()
dt = dt.reset_index(drop = True)

##############Get Summary Statistics###############
not_dup = dt["is_duplicate"].value_counts()[0] / dt["is_duplicate"].shape[0]
dup = 1 - not_dup
q1_len = []
q2_len = []
for i in range(len(dt)):
     q1_len.append(len(dt['question1'][i].split(" ")))
     q2_len.append(len(dt['question2'][i].split(" ")))
     
q_max = max(max(q1_len), max(q2_len))
q_min = min(min(q1_len), min(q2_len))

def argsort(seq):
     return sorted(range(len(seq)), key=seq.__getitem__)

q_max_index = argsort(q2_len)[len(q2_len) - 1] 
q_min_index =  argsort(q2_len)[0]

dt['question2'][q_max_index]
dt['question1'][q_max_index]
dt['is_duplicate'][q_max_index]
dt['question1'][q_min_index]
dt['question2'][q_min_index]
dt['is_duplicate'][q_min_index]

np.std(q1_len)
np.std(q2_len)
np.mean(q1_len)
np.mean(q2_len)
np.max(q1_len)
np.max(q2_len)
np.min(q1_len)
np.min(q2_len)

##############Some Exploratory Analysis###############
def build_ngram(n, stop_words = frozenset(stopwords.words('english'))):
     combined_questions = []
     cv = CountVectorizer(ngram_range = (n,n),\
               stop_words = stop_words)
     tk_func = cv.build_analyzer()
     for i in range(len(dt)):
               combined_questions  += tk_func(dt['question1'][i])
               if dt['is_duplicate'][i] == 0:
                    combined_questions += tk_func(dt['question2'][i])
     
          
     return combined_questions

unigram_str = ",".join(build_ngram(1))
unigram_str_stop = ",".join(build_ngram(1, None))

wechat_mask = np.array(Image.open("WeChat.png"))
ui_mask = np.array(Image.open("university.png"))
qq_mask = np.array(Image.open("qq.png"))
quora_mask = np.array(Image.open("quora.jpg"))

wc = WordCloud(background_color = "white", \
               max_words = 1000, mask = wechat_mask)

wc.generate(unigram_str)

rcParams['figure.figsize'] = 10, 15
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.figure()

cnt1 = cl.Counter(build_ngram(1))
cnt2 = cl.Counter(build_ngram(2))
cnt3 = cl.Counter(build_ngram(3))

unigram = cnt1.most_common(15)
bigram =  cnt2.most_common(15)
trigram = cnt3.most_common(15)

unigram_nostop = cl.Counter(build_ngram(1, None)).most_common(15)
bigram_nostop = cl.Counter(build_ngram(2, None)).most_common(15)
trigram_nostop = cl.Counter(build_ngram(3, None)).most_common(15)

unigram_values = [item[1] for item in unigram]
unigram_text = [item[0] for item in unigram]
bigram_values = [item[1] for item in bigram]
bigram_text = [item[0] for item in bigram]
trigram_values = [item[1] for item in trigram]
trigram_text = [item[0] for item in trigram]


unigram_values_nostop = [item[1] for item in unigram_nostop]
unigram_text_nostop = [item[0] for item in unigram_nostop]
bigram_values_nostop = [item[1] for item in bigram_nostop]
bigram_text_nostop = [item[0] for item in bigram_nostop]
trigram_values_nostop = [item[1] for item in trigram_nostop]
trigram_text_nostop = [item[0] for item in trigram_nostop]


rcParams['figure.figsize'] = 15, 10
sns.set(font_scale=2)  
fig, ax = plt.subplots()
sns.barplot(x = unigram_values, y = unigram_text)
plt.title("Most Common Unigrams(without stopwords)")
sns.barplot(x = bigram_values, y = bigram_text)
plt.title("Most Common Bigrams(without stopwords)")
sns.barplot(x = trigram_values, y = trigram_text)
plt.title("Most Common Trigrams(without stopwords)")

sns.barplot(x = unigram_values_nostop, y = unigram_text_nostop)
plt.title("Most Common Unigrams(with stopwords)")
sns.barplot(x = bigram_values_nostop, y = bigram_text_nostop)
plt.title("Most Common Bigrams(with stopwords)")
sns.barplot(x = trigram_values_nostop, y = trigram_text_nostop)
plt.title("Most Common Trigrams(with stopwords)")

##############Document Clustering##############

Stopwords = [set(stopwords.words('english') + get_stop_words('english'))]  
punctuation = string.punctuation+'``' + "''" + "..." 

def tokenize(text):
     tokens = [token for token in text.split() if token not in Stopwords[0]]
     tokens = word_tokenize(" ".join(tokens))
     tokens = [token for token in tokens if token not in punctuation]
     tokens = [token for token in tokens if token not in ["'s","n't","ca","doe"]]
     lemmer = WordNetLemmatizer()
     lemms = map(lemmer.lemmatize,tokens)
     return  lemms

cv = CountVectorizer(stop_words = "english",\
       lowercase = True, tokenizer = tokenize)
data = cv.fit_transform(dt['question1'])
terms = cv.get_feature_names()

##############K-means Clustering##############
k = 10
km = KMeans(n_clusters = k)
kmeans = km.fit(data)

centroids = kmeans.cluster_centers_.argsort()[:,::-1]

topic_word = np.zeros((10,10), dtype = object)
for i in range(k):
     for j in range(10):
          topic_word[i, j] = terms[centroids[i, :10][j]]


df = pd.DataFrame(topic_word)
df.to_csv("freq_word.csv")

##############Topic Modeling##############
##############Non-Negaive Matrix Factorization##############

nmf = NMF(n_components=k, max_iter = 1000).fit(data)

topic_word_nmf = np.zeros((10,10), dtype = object)

components = nmf.components_.argsort()[:,::-1]

for i in range(k):
     for j in range(10):
          topic_word_nmf[i, j] = terms[components[i, :10][j]]


df1 = pd.DataFrame(topic_word_nmf)
df1.to_csv("freq_word_nmf.csv")
##############Latent Dirichlet allocation###############

lda = LatentDirichletAllocation(n_topics = k, max_iter = 5,\
                               learning_method = "online", learning_offset=5.,\
                               random_state = 10).fit(data)

topic_word_lda = np.zeros((10,10), dtype = object)
components_lda = lda.components_.argsort()[:,::-1]

for i in range(k):
     for j in range(10):
          topic_word_lda[i, j] = terms[components_lda[i, :10][j]]


df2 = pd.DataFrame(topic_word_lda)
df2.to_csv("freq_word_lda.csv")

