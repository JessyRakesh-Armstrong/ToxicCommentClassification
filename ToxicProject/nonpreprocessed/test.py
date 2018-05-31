﻿import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import re, string
import csv
import sys

# Helper function to calculate log count ratio
def pr(y_i, y):
    p = train_vectors[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

# Create fitted SVM models using NB features
def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = train_vectors.multiply(r)
    return m.fit(x_nb, y), r

####Create labels and load CSV files into pandas DataFrames ###
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])
x_train = train
x_test = test

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

### td-idf word counts ###
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
train_vectors = vec.fit_transform(x_train['comment_text'].values.astype('U'))
test_vectors = vec.transform(x_test['comment_text'].values.astype('U'))

#### For each label, create NB-SVM model and predict using test data features ###
### into preds array ###
preds = np.zeros((len(x_test), len(labels)))
for i,j in enumerate(labels):
    print('fit',j)
    m,r = get_mdl(x_train[j])
    preds[:,i] = m.predict_proba(test_vectors.multiply(r))[:,1]

### Save results to CSV files ###
# preds, continuous values
new = pd.DataFrame(data=preds, columns=['toxic','severe_toxic','obscene','threat',
                                        'insult','identity_hate'])
if sys.argv[1] == 'train.csv':
    new.to_csv("preds.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
else:
    new.to_csv("p_preds.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)

# classified_preds, classification with threshold .5
classified_preds = np.zeros((len(x_test), len(labels)))
for row in range(len(preds)):
    for col in range(6):
        if preds[row][col] > .5:
            classified_preds[row][col] = 1
        else:
            classified_preds[row][col] = 0
new = pd.DataFrame(data=classified_preds, columns=['toxic','severe_toxic','obscene','threat',
                                        'insult','identity_hate'])
if sys.argv[1] == 'train.csv':
    new.to_csv("classified_preds.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
else:
    new.to_csv("p_classified_preds.csv", encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC, index=False)
    
### Accuracy outputs ###
test_data = x_test.drop(['id','comment_text'],axis=1)
test_data = test_data.values
acc = accuracy_score(test_data, classified_preds)
mse = mean_squared_error(test_data, classified_preds)
roc = roc_auc_score(test_data, classified_preds)
print('__________NB-SVM metrics__________')
print('Accuracy = ' + str(acc))
print('Mean square error = ' + str(mse))
print('ROC score = ' + str(roc))
