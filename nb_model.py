import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv('train.csv')

x_train, x_valid = train_test_split(train, test_size=.3)
vec=TfidfVectorizer()
tfidf_vectors = vec.fit_transform(x_train['comment_text'])
validation_vectors = vec.transform(x_valid['comment_text'])
x = tfidf_vectors
y = validation_vectors

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(x_valid), len(labels)))
for i,j in enumerate(labels):
    print('fit',j)
    m,r = get_mdl(x_train[j])
    preds[:,i] = m.predict_proba(validation_vectors.multiply(r))[:,1]

train_data = x_valid
train_data = train_data.drop(['id','comment_text'],axis=1)
train_data = train_data.values

print(mean_squared_error(train_data, preds))

#preds = np.zeros((len(train), len(labels)))
#for i, j in enumerate(labels):
#    print('fit', j)
#    m,r = get_mdl(train[j])
#    preds[:,i] = m.predict_proba(x.multiply(r))[:,1]

#train_data = train
#train_data.drop(['id','comment_text'], axis=1)
#train_data = train_data.values

#print explained_variance_score(train_data, preds)
