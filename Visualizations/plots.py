import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_self_sim_pretrain():
    df = pd.read_csv('pretrain.csv', index_col=0)
    df = df.rename(columns={'cpp': 'C++', 'c': 'C', 'python' : 'Python', 'js' : 'JavaScript', 'cuda': 'Cuda', 'julia': 'Julia', 'java' : 'Java', 'go': 'Go', 'fortran': 'Fortran', 'kotlin': 'Kotlin', 'scala': 'Scala', 'ruby': 'Ruby', 'html': 'HTML', 'lisp': 'Lisp', 'webassembly':'WebAssembly', 'elisp': 'ELisp', 'r':'R', 'assembly':'Assembly', 'mathematica':'Mathematica', 'cobol':'COBOL'})
    df = df.mask(df == 0)
    plot = sns.boxplot(df)
    plot.set(ylabel='Self-Similarity')
    fig = plot.get_figure()
    plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
    fig.tight_layout()
    fig.savefig('pretrain.png')

def plot_self_sim_no_pretrain():
    df = pd.read_csv('no_pretrain.csv', index_col=0)
    df = df.rename(columns={'cpp': 'C++', 'c': 'C', 'python' : 'Python', 'js' : 'JavaScript', 'cuda': 'Cuda', 'julia': 'Julia', 'java' : 'Java', 'go': 'Go', 'fortran': 'Fortran', 'kotlin': 'Kotlin', 'scala': 'Scala', 'ruby': 'Ruby', 'html': 'HTML', 'lisp': 'Lisp', 'webassembly':'WebAssembly', 'elisp': 'ELisp', 'r':'R', 'assembly':'Assembly', 'mathematica':'Mathematica', 'cobol':'COBOL'})
    df = df.mask(df == 0)
    plot = sns.boxplot(df)
    plot.set(ylabel='Self-Similarity')
    fig = plot.get_figure()
    plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
    fig.tight_layout()
    fig.savefig('box.png')

def plot_similarities_no_pretrain():
    sim_mat = np.load('sim_mat_no_pretrain.npy')
    langs = ['C++', 'Python', 'C', 'Cuda', 'Julia', 'Fortran', 'Java', 'Ruby', 'Kotlin', 'Scala', 'HTML', 'JavaScript', 'Assembly', 'Lisp', 'Go', 'COBOL', 'R', 'ELisp', 'WebAssembly', 'Mathematica']
    for i in range(20):
        sim_mat[i, i] = 0
    plot = sns.heatmap(sim_mat, xticklabels = langs, yticklabels = langs, cmap=sns.color_palette("Blues", as_cmap=True), vmax = 0.3, vmin = 0)
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig('sim_no_pretrain.png')

def plot_similarities_pretrained():
    sim_mat = np.load('sim_mat_pretrain.npy')
    langs = ['C++', 'Python', 'C', 'Cuda', 'Julia', 'Fortran', 'Java', 'Ruby', 'Kotlin', 'Scala', 'HTML', 'JavaScript', 'Assembly', 'Lisp', 'Go', 'COBOL', 'R', 'ELisp', 'WebAssembly', 'Mathematica']
    for i in range(20):
        sim_mat[i, i] = 0
    plot = sns.heatmap(sim_mat, xticklabels = langs, yticklabels = langs, cmap=sns.color_palette("Blues", as_cmap=True), vmax = 0.8, vmin = 0.5)
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig('sim_pretrain.png')




