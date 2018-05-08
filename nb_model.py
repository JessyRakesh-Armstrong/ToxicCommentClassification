import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv('train.csv')
train['none'] = 1-train[labels].max(axis=1)

vec=TfidfVectorizer()
trn_term_doc = vec.fit_transform(train['comment_text'])
x = trn_term_doc

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(train), len(labels)))
for i, j in enumerate(labels):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(x.multiply(r))[:,1]

#x = data['comment_text']
#CV = CountVectorizer()
#count = CV.fit_transform(x)
#tf_transformer = TfidfTransformer(use_idf=False).fit(count)
#tf = tf_transformer.transform(count)

#clf = MultinomialNB()
#category = ['toxic', 'severe_toxic', ]
#clf.fit(count,)
