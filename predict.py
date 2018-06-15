import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.models import model_from_json
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import roc_auc_score
#train = pd.read_csv('train.csv')
test = pd.read_csv('MergedTest.csv')


#train.isnull().any(),test.isnull().any()
test.isnull().any(),test.isnull().any()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
#y = train[list_classes].values
y2 = test[list_classes].values
#training_comments = train["comment_text"]
testing_comments = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(testing_comments))
#tokenized_training_comments = tokenizer.texts_to_sequences(training_comments)
tokenized_testing_comments = tokenizer.texts_to_sequences(testing_comments)
#tokenized_training_comments[:1]
tokenized_testing_comments[:1]
maxlen = 200
X_test = pad_sequences(tokenized_testing_comments, maxlen=maxlen)

#model from json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights
loaded_model.load_weights("model.h5")
print("Loaded model")

#evaluation
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print("Loss and Accuracy scores: ")
score = loaded_model.evaluate(X_te, y2, verbose=0)
print("Loss: ", score[0])
print("Accuracy: ", score[1])


#predict
print("Predicting")
y_pred = loaded_model.predict(X_test, batch_size=50, verbose=1)

roc_score = roc_auc_score(y2, y_pred)
print("ROC accuracy: ", roc_score)

#generate csv results file
submission = pd.DataFrame.from_dict({'id': test['id']})
for com_id, col in enumerate(list_classes):
    submission[col] = y_pred[:,com_id]
submission.to_csv('submission.csv', index=False)