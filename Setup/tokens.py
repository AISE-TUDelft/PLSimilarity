import json
import re
from transformers import RobertaTokenizerFast
import datasets

def remove_comments_c(f):
    f['content'] = re.sub("\/\*(\*(?!\/)|[^*])*\*\/", "", str(f['content']))
    f['content'] = re.sub("\\\\.*\\n", "", f['content'])
    return f
def remove_comments_py(f):
    f['content'] = re.sub("#.*\\n", "", str(f['content']))
    f['content'] = re.sub("\"{3}((.|\n)*?)\"{3}", "", f['content'])
    return f
def remove_comments_julia(f):
    f['content'] = re.sub("#=((.|\n)*?)=#", "", f['content'])
    f['content'] = re.sub("#.*\\n", "", str(f['content']))
    return f
def remove_comments_lisp(f):
    f['content'] = re.sub(";.*\\n", "", str(f['content']))
    return f
def remove_comments_fortran(f):
    f['content'] = re.sub("!.*\\n", "", str(f['content']))
    return f
def remove_comments_cobol(f):
    f['content'] = re.sub("\n((\s){6})[/\*].*\n", "\n", str(f['content']))
    return f
def remove_comments_html(f):
    f['content'] = re.sub("<!--((.|\n)*?)-->", "", str(f['content']))
    return f
def remove_comments_matlab(f):
    f['content'] = re.sub("%{((.|\n)*?)%}", "", str(f['content']))
    f['content'] = re.sub("%.*\\n", "", str(f['content']))
    return f
def remove_comments_webassembly(f):
    f['content'] = re.sub("\(;((.|\n)*);\)", "", str(f['content']))
    f['content'] = re.sub(";;.*\\n", "", str(f['content']))
    return f
def remove_comments_assembly(f):
    f['content'] = re.sub(";.*\\n", "", str(f['content']))
    return f
def remove_comments_ruby(f):
    f['content'] = re.sub("#.*\\n", "", str(f['content']))
    f['content'] = re.sub("=begin((.|\n)*)=end", "", str(f['content']))
    return f
def remove_comments_mathematica(f):
    f['content'] = re.sub("\(\*((.|\n)*?)\*\)", "", str(f['content']))
    return f

tokenizer = RobertaTokenizerFast.from_pretrained('microsoft/codebert-base-mlm')

dataset = {}
dataset['java'] = datasets.load_from_disk('java_eval')
dataset['java'] = dataset['java'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['kotlin'] = datasets.load_from_disk('kotlin_eval')
dataset['kotlin'] = dataset['kotlin'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['cpp'] = datasets.load_from_disk('cpp_eval')
dataset['cpp'] = dataset['cpp'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['python'] = datasets.load_from_disk('python_eval')
dataset['python'] = dataset['python'].map(remove_comments_py, num_proc = os.cpu_count())
dataset['go'] = datasets.load_from_disk('go_eval')
dataset['go'] = dataset['go'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['julia'] = datasets.load_from_disk('julia_eval')
dataset['julia'] = dataset['julia'].map(remove_comments_julia, num_proc = os.cpu_count())
dataset['lisp'] = datasets.load_from_disk('lisp_eval')
dataset['lisp'] = dataset['lisp'].map(remove_comments_lisp, num_proc = os.cpu_count())
dataset['elisp'] = datasets.load_from_disk('elisp_eval')
dataset['elisp'] = dataset['elisp'].map(remove_comments_lisp, num_proc = os.cpu_count())
dataset['scala'] = datasets.load_from_disk('scala_eval')
dataset['scala'] = dataset['scala'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['fortran'] = datasets.load_from_disk('fortran_eval')
dataset['fortran'] = dataset['fortran'].map(remove_comments_fortran, num_proc = os.cpu_count())
dataset['cobol'] = datasets.load_from_disk('cobol_eval')
dataset['cobol'] = dataset['cobol'].map(remove_comments_cobol, num_proc = os.cpu_count())
dataset['cuda'] = datasets.load_from_disk('cuda_eval')
dataset['cuda'] = dataset['cuda'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['c'] = datasets.load_from_disk('c_eval')
dataset['c'] = dataset['c'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['html'] = datasets.load_from_disk('html_eval')
dataset['html'] = dataset['html'].map(remove_comments_html, num_proc = os.cpu_count())
dataset['js'] = datasets.load_from_disk('js_eval')
dataset['js'] = dataset['js'].map(remove_comments_c, num_proc = os.cpu_count())
dataset['r'] = datasets.load_from_disk('r_eval')
dataset['r'] = dataset['r'].map(remove_comments_py, num_proc = os.cpu_count())
dataset['mathematica'] = datasets.load_from_disk('mathematica_eval')
dataset['mathematica'] = dataset['mathematica'].map(remove_comments_mathematica, num_proc = os.cpu_count())
dataset['webassembly'] = datasets.load_from_disk('webassembly_eval')
dataset['webassembly'] = dataset['webassembly'].map(remove_comments_webassembly, num_proc = os.cpu_count())
dataset['assembly'] = datasets.load_from_disk('assembly_eval')
dataset['assembly'] = dataset['assembly'].map(remove_comments_assembly, num_proc = os.cpu_count())
dataset['ruby'] = datasets.load_from_disk('ruby_eval')
dataset['ruby'] = dataset['ruby'].map(remove_comments_ruby, num_proc = os.cpu_count())


for ds in dataset:
    dataset[ds] = dataset[ds].map(tokenize, batched = True, num_proc = 16)

ids = {}
for lang in dataset:
    ids[lang] = {}
    for i, f in enumerate(tqdm(dataset[lang])):
        for t in f['input_ids']:
            if t not in ids[lang]:
                ids[lang][t] = 1
            else:
                ids[lang][t] += 1

with open('token_counts.json', 'w') as outfile:
    outfile.write(json.dumps(ids))
