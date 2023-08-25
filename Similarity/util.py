import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm

def cosim(a, b):
    return dot(a,b)/(norm(a)*norm(b))

def lang_similarity(a, b):
    tokens_a = dataset[a]
    tokens_b = dataset[b]

    token_sim = {}
    for token in tokens_a.keys():
        token_sim[token] = sorted_token_similarity(tokens_a[token], tokens_b[token])
    return token_sim.values()

def lang_similarity_multi(a, b):
    tokens_a = a
    tokens_b = b
    
    inlist = []
    for t in tokens_a:
        inlist.append((tokens_a[t], tokens_b[t]))
    
    res_list = []
    p = Pool(4)
    with tqdm(total = len(inlist)) as pbar:
        for i, res in enumerate(p.imap_unordered(multi_similarity, inlist)):
            pbar.update()
            res_list.append(res)
        pbar.close()
        p.close()
        p.join()
    return np.mean(res_list)

def multi_similarity(intuple):
    return np.mean(sorted_token_similarity(intuple[0], intuple[1]))

def sorted_token_similarity(a, b):
    sims = np.ndarray(shape=(len(a), len(b)))
    for i, t1 in enumerate(a):
        for j, t2 in enumerate(b):
            sims[i][j] = cosim(t1,t2)
    inf_sims = []
    r = min(len(a), len(b))
    for i in range(r):
        m = sims.max()
        loc = np.where(sims == m)
        x = loc[0][0]
        y = loc[1][0]
        sims[x,:] = 0
        sims[:,y] = 0
        inf_sims.append(m)
    return inf_sims

def self_similarity(a):
    sims = np.ndarray(shape=(len(a), len(a)))
    for i, t1 in enumerate(a):
        for j, t2 in enumerate(a):
            if i == j:
                continue
            sims[i][j] = cosim(t1,t2)
    inf_sims = []
    r = len(a)
    for i in range(r):
        m = sims.max()
        loc = np.where(sims == m)
        x = loc[0][0]
        y = loc[1][0]
        sims[x,:] = 0
        sims[:,y] = 0
        inf_sims.append(m)
    return inf_sims

def self_similarity_lang(lang):
    sims = []
    for t in lang:
        sims.append(self_similarity(lang[t]))
    return sims
