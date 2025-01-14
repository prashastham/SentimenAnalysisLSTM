import marimo

__generated_with = "0.10.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from nltk.corpus import stopwords 
    from collections import Counter
    import string
    import re
    import seaborn as sns
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    return (
        Counter,
        DataLoader,
        F,
        TensorDataset,
        mo,
        nn,
        np,
        pd,
        plt,
        re,
        sns,
        stopwords,
        string,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell
def _():
    import nltk
    nltk.download('stopwords')
    return (nltk,)


@app.cell
def _(torch):
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        device = torch.device("cuda")
        print("GPU is available")
    elif mps_available:
        device = torch.device("mps")
        print("MPS are available")
    else:
        device = torch.device("cpu")
        print("Acceleration not available")
    return cuda_available, device, mps_available


@app.cell
def _(pd):
    imdb_csv = 'data/IMDB Dataset.csv'
    df = pd.read_csv(imdb_csv)
    df.head()
    return df, imdb_csv


@app.cell
def _(df):
    len(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """

        ###Splitting to train and test data

        We will split data to train and test initially, to avoid data lekage (make sure the test/train data is segregated).

        """
    ).callout()
    return


@app.cell
def _(df, train_test_split):
    X,y = df['review'].values,df['sentiment'].values
    x_train,x_test,y_train,y_test = train_test_split(X,y,stratify=y)
    print(f'shape of train data is {x_train.shape}')
    print(f'shape of test data is {x_test.shape}')
    return X, x_test, x_train, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""###Analysing senitment: Is the data set balanced?""")
    return


@app.cell
def _(np, pd, plt, sns, y_train):
    dd = pd.Series(y_train).value_counts()
    ax = sns.barplot(x=np.array(['negative','positive']),y=dd.values, hue=['negative','positive'])
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Frequencey')
    plt.show()
    return ax, dd


@app.cell
def _(Counter, np, re, stopwords):
    def preprocess_string(s):
        # Remove all non-word characters (everything except numbers and letters)
        s = re.sub(r"[^\w\s]", '', s)
        # Replace all runs of whitespaces with no space
        s = re.sub(r"\s+", '', s)
        # replace digits with no space
        s = re.sub(r"\d", '', s)

        return s

    def tockenize(x_train,y_train,x_val,y_val):
        word_list = []

        stop_words = set(stopwords.words('english')) 
        for sent in x_train:
            for word in sent.lower().split():
                word = preprocess_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)

        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

        # tockenize
        final_list_train,final_list_test = [],[]
        for sent in x_train:
                final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                         if preprocess_string(word) in onehot_dict.keys()])
        for sent in x_val:
                final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                        if preprocess_string(word) in onehot_dict.keys()])

        encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test),  np.array(encoded_test), onehot_dict
    return preprocess_string, tockenize


@app.cell
def _(tockenize, x_test, x_train, y_test, y_train):
    X_train,Y_train,X_test,Y_test,vocab = tockenize(x_train,y_train,x_test,y_test)
    return X_test, X_train, Y_test, Y_train, vocab


if __name__ == "__main__":
    app.run()
