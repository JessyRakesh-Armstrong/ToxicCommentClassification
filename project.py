import sys, os, re, csv, codecs, numpy as np, pandas as pd
import gensim.models.keyedvectors as word2vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


embedding_type = input("Input embedding type: normal, word2vec or fasttext\n")
print(embedding_type)

print("test")

train = pd.read_csv('train.csv')
test = pd.read_csv('MergedTest.csv')


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
y2 = test[list_classes].values
training_comments = train["comment_text"]
testing_comments = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(training_comments))
tokenized_training_comments = tokenizer.texts_to_sequences(training_comments)
tokenized_testing_comments = tokenizer.texts_to_sequences(testing_comments)
list_tokenized_train[:1]

maxlen = 200
X_train = pad_sequences(tokenized_training_comments, maxlen=maxlen)
X_test = pad_sequences(tokenized_testing_comments, maxlen=maxlen)

inp = Input(shape=(maxlen, )) 


#embedding references:
#FASTTEXT
#https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
#@article{bojanowski2016enriching,
#title={Enriching Word Vectors with Subword Information},
#author={Bojanowski, Piotr and Grave, Edouard and Joulin, Armand and Mikolov, Tomas},
#journal={arXiv preprint arXiv:1607.04606},
#year={2016}
#}
#
#WORD2VEC
#https://github.com/3Top/word2vec-api
#

if embedding_type == "word2vec":
	embed_size = 300
	word2vecfile = word2vec.KeyedVectors.load_word2vec_format("glove.6b.300d", binary=True)
	embeddings_index = dict()
	for word in word2vecfile.wv.vocab:
		embeddings_index[word] = word2vecDict.word_vec(word)  
elif embedding_type == "fasttext":
	embed_size = 300
	embeddings_index = dict()
	f = open("wiki.en.vec")
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
elif embedding_type == "normal":
	embed_size = 128

all_embs = np.stack(list(embeddings_index.values()))
emb_mean,emb_std = all_embs.mean(), all_embs.std()
nb_words = len(tokenizer.word_index)
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector   
del(embeddings_index)



#embed_size = 128
if embedding_type == "normal":
	x = Embedding(max_features, embed_size)(inp)
else:
	x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
batch_size = 50
epochs = 2
model.fit(X_train,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
scores = model.evaluate(X_train, y, verbose=0)


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model")

#print("Predicting")
#y_pred = model.predict(X_test, batch_size=1024)

#submission = pd.DataFrame.from_dict({'id': test['id']})
#for idx, col in enumerate(list_classes):
#    submission[col] = y_pred[:,idx]
#submission.to_csv('submission.csv', index=False)

#roc_score = roc_auc_score(y2, y_pred)
#print("ROC accuracy: ", roc_score)
