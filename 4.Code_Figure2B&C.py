# -*- coding: utf-8 -*-
"""
This is the code to prepare data for Figure2B&C 

"""

from gensim.models import Word2Vec
import pandas as pd
import random
import igraph as ig
import numpy as np

## load pre-trained w2v model
model = Word2Vec.load("word2vec_hk_2022.model") 

## load the created dictionary: uncivil_words (N=1956)
uncivil_words = [x.strip() for x in open('uncivil_words.txt',encoding="utf8").read().split('\n')]

## load uncivil words with word frequency/percentage in the corpus        
df = pd.read_csv('topUncivil.txt', sep='\t')
vocab = model.wv.key_to_index # all words
ucvwd = [x for x in uncivil_words if x in vocab] # max 1911/1956 words in w2v
df.p[df.words.isin(ucvwd)].sum() # max = 0.9574114069984


## the coverage by random sampling seeds
def random_seeds(n_sds):    
    for j in range(10):
        seed_words = random.sample(ucvwd, n_sds) # random sample from the dictionary
        all_collected = seed_words
        i = 0
        while(len(seed_words)>0):
            new_words = pd.DataFrame()
            for word in seed_words:
                try:
                    nw = model.wv.most_similar(word,topn=20)
                    nw = pd.DataFrame(nw,columns=["words","similarity"])
                    nw['w'] = word
                    new_words = new_words.append(nw)
                except:
                    print (word)
            
            x=set(new_words.words)
            matching = [s for s in x if any(xs in s for xs in ucvwd)]
            seed_words = [w for w in matching if w not in all_collected]
            all_collected = list(set(all_collected + seed_words))
            
            done = []
            for a in ucvwd:
                for b in all_collected:
                    if a in b:
                        done.append(a)    
            p = df.p[df.words.isin(done)].sum()
            
            i = i+1
            
            with open('coverage_'+str(n_sds)+'.txt', 'a') as f:
                f.write(str(j)+'\t'+str(i)+'\t'+str(len(all_collected))+'\t'+str(len(set(done)))+'\t'+str(p)+'\n')
            # step + uncivil words found + # of words in our dictionary + % of the words adjusted by frequency
            print(str(j)+'\t'+str(i)+'\t'+str(len(all_collected))+'\t'+str(len(set(done)))+'\t'+str(p))
    
## sample 10/50/100 respectively (it may take a long time)
random_seeds(10)
random_seeds(50)
random_seeds(100)


## create a network (nodes are the uncivil words, edges are the topN similarity)
## if it is strongly connected, starting from any seeds will eventually get the full dictionary
sim_words = pd.DataFrame()
for word in ucvwd:
    nw = model.wv.most_similar(word,topn=20)
    nw = pd.DataFrame(nw,columns=["words","similarity"])
    nw['w'] = word
    sim_words = sim_words.append(nw)

sim_network = sim_words[sim_words.words.isin(ucvwd)]    
sim_network = sim_network[['w','words']]
G = ig.Graph.TupleList(sim_network.values, weights=False, directed=True)
cpt = G.clusters(mode='strong')
np.max(cpt.sizes())

