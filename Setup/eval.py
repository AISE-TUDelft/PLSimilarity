import os
from transformers import AutoTokenizer, RobertaForMaskedLM, pipeline, Trainer, TrainingArguments
import datasets
import evaluate
import torch
import json
from tqdm import tqdm
import re

from datasets import disable_caching

disable_caching()

#Put correct models, languages, and dataset here
eval_lang = 'lisp'

model = RobertaForMaskedLM.from_pretrained('./outputs/' + eval_lang)
device = torch.device('cuda:0')
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('codeBERTScoreTokenizer')

dataset = {}

dataset['lisp'] = datasets.load_dataset('<Dataset name>', split = 'train[:10%]')

def tokenize(f):
    return tokenizer(f['content'])

for ds in dataset:
    dataset[ds] = dataset[ds].map(tokenize, batched = True, num_proc = os.cpu_count()//2)
#Generate this file first
import json
with open('token_counts.json', 'r') as infile:  
    ids = json.loads(infile.read())

from multiprocessing import Pool

common_ids = []
for i in range(1, 50266):
    keep = True
    for lang in ids:
        if str(i) not in ids[lang]:
            keep = False
            break
    if keep:
        common_ids.append(i)
    


def get_common_ids_in_file(in_tuple):
    file = in_tuple
    return_list = []
    for token in common_ids:
        if token in file['input_ids']:
            return_list.append(token)

    return (file['id'],return_list)

file_id = list(range(len(dataset[eval_lang])))
dataset[eval_lang] = dataset[eval_lang].add_column('id', file_id)

inlist = dataset[eval_lang]

res_list = []
p = Pool(100) #Most will probably need to reduce the number of threads
with tqdm(total = len(inlist)) as pbar:
    for i, res in enumerate(p.imap_unordered(get_common_ids_in_file, inlist)):
        pbar.update()
        res_list.append(res)
    pbar.close()
    p.close()
    p.join()


candidates_by_token = {}
candidates_by_token[eval_lang] = {}
for t in common_ids:
    candidates_by_token[eval_lang][t] = []
for res in tqdm(res_list):
    file_id = res[0]
    tokens = res[1]
    for t in tokens:
        candidates_by_token[eval_lang][t].append(file_id)

import random
to_sample = {}
to_sample[eval_lang] = {}
for t in tqdm(common_ids):
    if len(candidates_by_token[eval_lang][t]) > 50:
        to_sample[eval_lang][t] = random.sample(candidates_by_token[eval_lang][t], 50)
    else:
        to_sample[eval_lang][t] = candidates_by_token[eval_lang][t]

def gen_samples(in_tuple):
    files = in_tuple[0]
    token = in_tuple[1]
    return_list = []
    for file in files:
        occurances = [i for i, x in enumerate(file) if x == token]
        x = random.sample(occurances, 1)[0]
        file[x] = 50264 #Mask token
        index = min(x, 256)
        return_list.append((token, file[x-256:x+256], index))

    return return_list

res_list = []
inlist = []
data = dataset[eval_lang]['input_ids']
for t in tqdm(to_sample[eval_lang]):
    inlist.append(([data[i] for i in to_sample[eval_lang][t]], t))

p = Pool(100)
with tqdm(total = len(inlist)) as pbar:
    for i, res in tqdm(enumerate(p.imap_unordered(gen_samples, inlist))):
        pbar.update()
        res_list.append(res)
    pbar.close()
    p.close()
    p.join()

samples = {}
samples[eval_lang] = {}
for i in common_ids:
    samples[eval_lang][i] = []
for i in res_list:
    for s in i:
        samples[eval_lang][s[0]].append((s[1],s[2]))

def batch_iterator(l, n= 5):
    out_list_indices = []
    out_list_samples = []
    for i, tup in enumerate(l):
        out_list_samples.append(tup[0] + [tokenizer.pad_token_id] * (512-len(tup[0])))
        out_list_indices.append(tup[1])
        if i % n == 0 and i > 0:
            yield out_list_indices, out_list_samples
            out_list_indices = []
            out_list_samples = []
    yield out_list_indices, out_list_samples
    

samples_encoded = {}
samples_encoded[eval_lang] = {}
for i in common_ids:
    samples_encoded[eval_lang][i] = []
for token in tqdm(samples[eval_lang]):
    samples_encoded[eval_lang][token] = []
    for batch in batch_iterator(samples[eval_lang][token]):
        mask_indices = batch[0]
        inp = torch.tensor(batch[1]).to(device)
        if len(batch[0]) > 0:
            out = model(inp, output_hidden_states = True)
            for i, ind in enumerate(mask_indices):
                samples_encoded[eval_lang][token].append(out.hidden_states[-1][i][ind].detach().cpu().numpy().tolist())

with open('results_' + eval_lang +'.json', 'w') as outfile:
    outfile.write(json.dumps(samples_encoded))                

