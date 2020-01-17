# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:19:59 2019

@author: Nika.Khvedelidze
"""

import re
import pandas as pd
from time import time
from collections import defaultdict

import spacy

import logging

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

data = pd.read_csv('simpsons_dataset.csv')
data.shape
data.isnull().sum()
data = data.dropna().reset_index(drop=True)
data.isnull().sum()

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    
    if len(txt)>2:
        return ' '.join(txt)
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in data['spoken_words'])


t = time()
txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins.' .format(round((time()-t)/60, 2)))

data_clean=pd.DataFrame({'clean':txt})
data_clean = data_clean.dropna().drop_duplicates()
data_clean.shape

from gensim.models.phrases import Phrases, Phraser

sent = [row.split() for row in data_clean['clean']]
phrases = Phrases(sent, min_count=30, progress_per=10000)

bigram=Phraser(phrases)
sentences = bigram[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

sorted(word_freq, key=word_freq.get, reverse=True)[:10]

import multiprocessing

from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

t=time()
w2v_model.build_vocab(sentences, progress_per=10000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

w2v_model.wv.most_similar(positive=["homer"])

w2v_model.wv.most_similar(positive=["homer_simpson"])

w2v_model.wv.most_similar(positive=["marge"])


w2v_model.wv.most_similar(positive=["bart"])

w2v_model.wv.similarity('maggie', 'baby')

w2v_model.wv.similarity('bart', 'nelson')

w2v_model.wv.doesnt_match(['jimbo', 'milhouse', 'kearney'])

w2v_model.wv.doesnt_match(["nelson", "bart", "milhouse"])

w2v_model.wv.doesnt_match(['homer', 'patty', 'selma'])

w2v_model.wv.most_similar(positive=["woman", "homer"], negative=["marge"], topn=3)

w2v_model.wv.most_similar(positive=["woman", "bart"], negative=["man"], topn=3)


