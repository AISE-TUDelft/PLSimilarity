from tqdm import tqdm
import numpy as np
import orjson
from util import lang_similarity_multi, self_similarity_lang

def lang_sim_pretrained():
    langs = ['results_pretrained_cpp.json',
    'results_pretrained_c.json',
    'results_pretrained_python.json',
    'results_pretrained_js.json',
    'results_pretrained_cuda.json',
    'results_pretrained_julia.json',
    'results_pretrained_java.json',
    'results_pretrained_go.json',
    'results_pretrained_fortran.json',
    'results_pretrained_kotlin.json',
    'results_pretrained_scala.json',
    'results_pretrained_ruby.json',
    'results_pretrained_html.json',
    'results_pretrained_lisp.json',
    'results_pretrained_webassembly.json',
    'results_pretrained_elisp.json',
    'results_pretrained_r.json',
    'results_pretrained_assembly.json',
    'results_pretrained_mathematica.json',
    'results_pretrained_cobol.json',
    ]
    dataset = {}
    for l in tqdm(langs):
        with open('results/pretrained/' + l, 'r') as infile:
            data = orjson.loads(infile.read())
        lang = l.replace('results_pretrained_', '')
        lang = lang.replace('.json', '')
        for token in data[lang]:
            data[lang][token] = np.array(data[lang][token])
        dataset[lang] = data[lang]

    langs = list(dataset.keys())
    sim_mat = np.empty(shape=(len(langs),len(langs)))

    for i in tqdm(range(len(langs))):
        sim_mat[i,i] = 0
        for j in range(i+1, len(langs)):
            sim_mat[i,j] = lang_similarity_multi(dataset[langs[i]], dataset[langs[j]])
            sim_mat[j,i] = sim_mat[i, j]
    np.save('sim_mat_pretrained', sim_mat)

def lang_sim_no_pretrain():
    langs = ['results_no_pretrained_cpp.json',
    'results_no_pretrained_python.json',
    'results_no_pretrained_c.json',
    'results_no_pretrained_julia.json',
    'results_no_pretrained_cuda.json',
    'results_no_pretrained_fortran.json',
    'results_no_pretrained_java.json',
    'results_no_pretrained_ruby.json',
    'results_no_pretrained_kotlin.json',
    'results_no_pretrained_scala.json',
    'results_no_pretrained_html.json',
    'results_no_pretrained_js.json',
    'results_no_pretrained_assembly.json',
    'results_no_pretrained_lisp.json',
    'results_no_pretrained_go.json',
    'results_no_pretrained_cobol.json',
    'results_no_pretrained_r.json',
    'results_no_pretrained_elisp.json',
    'results_no_pretrained_webassembly.json',
    'results_no_pretrained_mathematica.json',
    ]
    dataset = {}
    for l in tqdm(langs):
        with open('results/no_pretrained/' + l, 'r') as infile:
            data = orjson.loads(infile.read())
        lang = l.replace('results_no_pretrained_', '')
        lang = lang.replace('.json', '')
        for token in data[lang]:
            data[lang][token] = np.array(data[lang][token])
        dataset[lang] = data[lang]

    langs = list(dataset.keys())
    sim_mat = np.empty(shape=(len(langs),len(langs)))

    for i in tqdm(range(len(langs))):
        sim_mat[i,i] = 0
        for j in range(i+1, len(langs)):
            sim_mat[i,j] = lang_similarity_multi(dataset[langs[i]], dataset[langs[j]])
            sim_mat[j,i] = sim_mat[i, j]
    np.save('sim_mat_no_pretrained', sim_mat)

def self_sim_pretrain():
    langs = ['results_pretrained_cpp.json',
    'results_pretrained_c.json',
    'results_pretrained_python.json',
    'results_pretrained_js.json',
    'results_pretrained_cuda.json',
    'results_pretrained_julia.json',
    'results_pretrained_java.json',
    'results_pretrained_go.json',
    'results_pretrained_fortran.json',
    'results_pretrained_kotlin.json',
    'results_pretrained_scala.json',
    'results_pretrained_ruby.json',
    'results_pretrained_html.json',
    'results_pretrained_lisp.json',
    'results_pretrained_webassembly.json',
    'results_pretrained_elisp.json',
    'results_pretrained_r.json',
    'results_pretrained_assembly.json',
    'results_pretrained_mathematica.json',
    'results_pretrained_cobol.json',
    ]
    dataset = {}
    for l in tqdm(langs):
        with open('results/pretrained/' + l, 'r') as infile:
            data = orjson.loads(infile.read())
        lang = l.replace('results_pretrained_', '')
        lang = lang.replace('.json', '')
        for token in data[lang]:
            data[lang][token] = np.array(data[lang][token])
        dataset[lang] = data[lang]

    sims = {}
    for lang in tqdm(dataset):
        sims[lang] = self_similarity_lang(dataset[lang])
        sims[lang] = [np.mean([min(1, x) for x in y]) for y in sims[lang]]
    df = pd.DataFrame.from_dict(sims)
    df.to_csv('no_pretrained.csv')



def self_sim_no_pretrain():
    langs = ['results_no_pretrained_cpp.json',
    'results_no_pretrained_python.json',
    'results_no_pretrained_c.json',
    'results_no_pretrained_julia.json',
    'results_no_pretrained_cuda.json',
    'results_no_pretrained_fortran.json',
    'results_no_pretrained_java.json',
    'results_no_pretrained_ruby.json',
    'results_no_pretrained_kotlin.json',
    'results_no_pretrained_scala.json',
    'results_no_pretrained_html.json',
    'results_no_pretrained_js.json',
    'results_no_pretrained_assembly.json',
    'results_no_pretrained_lisp.json',
    'results_no_pretrained_go.json',
    'results_no_pretrained_cobol.json',
    'results_no_pretrained_r.json',
    'results_no_pretrained_elisp.json',
    'results_no_pretrained_webassembly.json',
    'results_no_pretrained_mathematica.json',
    ]
    dataset = {}
    for l in tqdm(langs):
        with open('results/no_pretrained/' + l, 'r') as infile:
            data = orjson.loads(infile.read())
        lang = l.replace('results_no_pretrained_', '')
        lang = lang.replace('.json', '')
        for token in data[lang]:
            data[lang][token] = np.array(data[lang][token])
        dataset[lang] = data[lang]


    sims = {}
    for lang in tqdm(dataset):
        sims[lang] = self_similarity_lang(dataset[lang])
        sims[lang] = [np.mean([min(1, x) for x in y]) for y in sims[lang]]
    df = pd.DataFrame.from_dict(sims)
    df.to_csv('pretrained.csv')



