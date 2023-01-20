# -*- coding: utf-8 -*-
"""
"""

from gensim.models.keyedvectors import KeyedVectors
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string
from math import sqrt

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

from sklearn.cluster import KMeans
from itertools import repeat

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Concatenate, concatenate
from keras.callbacks import Callback
from keras.regularizers import l2, l1
from keras.layers import LeakyReLU
from keras.models import load_model
import os

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


######################## read data ################################
headlines = np.load('path/headline_abstract-data.npy')
labels = np.load('path/headline_abstract-labels.npy')
print("Headline shape: "+str(headlines.shape))
print("Label shape: "+str(labels.shape))

########################## pre-processing #######################
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

Tokens = []
finalTokens =[]
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english')) 
for i in range(len(headlines)):
    tempTokens = headlines[i].lower() #converting to lower case
    tempTokens = tempTokens.translate(str.maketrans('','',"~@#$%^&*()_-+={}[]|\/><'.,-+`:;1234567890"))
    tempTokens = tokenizer.tokenize(tempTokens) #tokenization 
    
#    for j in range(len(tempTokens)):
#        tempTokens[j] = lemmatizer.lemmatize(tempTokens[j] , get_wordnet_pos(tempTokens[j] )) #lemetization
        
    tempTokensStopRemoval = [word for word in tempTokens if word not in stop_words] #stopword removal 
    Tokens.append(tempTokens) # tokens with out stopword removal 
    finalTokens.append(tempTokensStopRemoval) # tokens after stopword removal

# De-tokenized sentances
deTokenized = []
for j in range(len(finalTokens)):
    tempTokens = []
    tempDetoken = finalTokens[j]
    tempDetoken = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tempDetoken]).strip()
    deTokenized.append(tempDetoken)

tokenised =  finalTokens

################################ train test val split ######################
x_train_val, x_test, y_train_val, y_test = train_test_split(deTokenized, labels, test_size=0.20, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)
print("x_train: "+str(len(x_train)))
print("y_train: "+str(y_train.shape))

print("x_val: "+str(len(x_val)))
print("y_val: "+str(y_val.shape))

print("x_test: "+str(len(x_test)))
print("y_test: "+str(y_test.shape))

###########  Parameters Splitings for CNN Model  ############
#EMBEDDING_DIM=100 # how big is each word vector
max_features = 39088 #3286 #13260  how many unique words to use (i.e num rows in embedding vector)
maxlen = 30 # max number of words in a comment to use

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(deTokenized)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_val = tokenizer.texts_to_sequences(x_val)
sequences_test=tokenizer.texts_to_sequences(x_test)

x_train_padded = pad_sequences(sequences_train,padding='post', maxlen=maxlen)
x_val_padded = pad_sequences(sequences_val,padding='post', maxlen=maxlen)
x_test_padded = pad_sequences(sequences_test,padding='post', maxlen=maxlen)
word_index = tokenizer.word_index

##########  Glove EMbedding for LSTM Model  ############

embeddings_index_glove = {}
f = open(os.path.join('path', 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index_glove[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index_glove))


print("Embedding Glove Strted")
embedding_dim_glove = 50 # how big is each word vector
vocabulary_size = min(max_features, len(word_index)) + 1

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix_glove = np.zeros((vocabulary_size, embedding_dim_glove))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector_glove = embeddings_index_glove.get(word)
    if embedding_vector_glove is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix_glove[i] = embedding_vector_glove
    else:
        # doesn't exist, assign a random vector
        embedding_matrix_glove[i] = np.random.randn(embedding_dim_glove)

print("Embedding GLove END")

##########  SSWE EMbedding for LSTM Model  ############
embeddings_index_sswe = {}
f = open(os.path.join('path', 'sswe-u.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index_sswe[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index_sswe))

print("Embedding SSWE Strted")
embedding_dim_sswe = 50 # how big is each word vector
vocabulary_size = min(max_features, len(word_index)) + 1

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix_sswe = np.zeros((vocabulary_size, embedding_dim_sswe))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector_sswe = embeddings_index_sswe.get(word)
    if embedding_vector_sswe is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix_sswe[i] = embedding_vector_sswe
    else:
        # doesn't exist, assign a random vector
        embedding_matrix_sswe[i] = np.random.randn(embedding_dim_sswe)
print("Embedding SSWE END")

##########  Dual LSTM Model  ############
sentence_indices_glove = Input(name='sentence_indices_glove',shape=[maxlen], dtype='int32')
embeddings_glove = Embedding(vocabulary_size, embedding_dim_glove, weights=[embedding_matrix_glove],trainable = False)(sentence_indices_glove)
X_glove = LSTM(64,
         activation='relu', 
         kernel_regularizer=l2(0.001), 
         recurrent_regularizer=l2(0.001),
         )(embeddings_glove)
X_glove = Dropout(0.5)(X_glove) 

sentence_indices_sswe = Input(name='sentence_indices_sswe',shape=[maxlen], dtype='int32')
embeddings_sswe= Embedding(vocabulary_size, embedding_dim_sswe, weights=[embedding_matrix_sswe],trainable = False)(sentence_indices_sswe)
X_sswe = LSTM(64,
         activation='relu', 
         kernel_regularizer=l2(0.001), 
         recurrent_regularizer=l2(0.001),
         )(embeddings_sswe) 
X_sswe = Dropout(0.5)(X_sswe) 

finalModel = keras.layers.concatenate([X_glove, X_sswe])
outLeaky = Dense(64) (finalModel)
outLeaky = LeakyReLU()(outLeaky) #alpha=0.3
out =  Dense(5, activation='softmax')(outLeaky)

model = Model(inputs=[sentence_indices_glove, sentence_indices_sswe], outputs=[out])
opt = keras.optimizers.Adam(lr=0.0005) 
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
checkpointer = ModelCheckpoint(filepath='file.h5', verbose=1, save_best_only=True)
history = model.fit([x_train_padded, x_train_padded], y_train, validation_data=([x_val_padded, x_val_padded],y_val), epochs=100, batch_size=64, callbacks=[checkpointer], verbose=1)
model.summary()

####################### Plot training & validation loss values ####################
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

####################### Evaluation1: RMSE ######################
#del model
best_model = load_model('file.h5')
predict_test = best_model.predict([x_test_padded,x_test_padded])
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

predict_test[0:2]

###########  Evaluation2: Acc@N, N == 1, 2, 3 ############
#Acc@N : N==1, 2, 3
maxEmoPredict_test = np.argmax(predict_test,1)
sortdMaxEmoActual_test = np.argsort(-y_test, axis=1)

#Acc@1
sumAT1_test = np.sum(maxEmoPredict_test == sortdMaxEmoActual_test[:,0]) 
accAT1_test = sumAT1_test / np.size(y_test,0)
print("Acc@1_test = "+ str(accAT1_test))

###########  Evaluation 3: APdocument  ############
X = predict_test # Predicted Labels
Y = y_test # Y --> Actual Labels 
#Result Matrix
APDocmatrix = np.zeros((len(X), 12))

xMean = np.mean(X,axis=1)
APDocmatrix[:,0] = xMean  #xMean

yMean = np.mean(Y,axis=1)
APDocmatrix[:,1] = yMean # yMean

XE0 = (X[:,0] - xMean) * (Y[:,0] - yMean) #(X1i - xMean) * (Y1i - yMean)
APDocmatrix[:,2] = XE0  

XE1 = (X[:,1] - xMean) * (Y[:,1] - yMean) #(X2i - xMean) * (Y2i - yMean)
APDocmatrix[:,3] = XE1

XE2 = (X[:,2] - xMean) * (Y[:,2] - yMean) #(X3i - xMean) * (Y3i - yMean)
APDocmatrix[:,4] = XE2

XE3 = (X[:,3] - xMean) * (Y[:,3] - yMean) #(X4i - xMean) * (Y4i - yMean)
APDocmatrix[:,5] = XE3

XE4 = (X[:,4] - xMean) * (Y[:,4] - yMean) #(X5i - xMean) * (Y5i - yMean)
APDocmatrix[:,6] = XE4

sigmaX = np.std(X, axis= 1) #standerd Deviation of X
APDocmatrix[:,7] = sigmaX

sigmaY = np.std(Y, axis= 1) #standerd deviation of Y
APDocmatrix[:,8] = sigmaY

emoLen = X.shape[1]
denominator = (emoLen - 1) *(sigmaX) * (sigmaY) #Denominator 
APDocmatrix[:,9] = denominator

numerator = np.sum(APDocmatrix[:,2:7], axis = 1)  #Numerator 
APDocmatrix[:,10] = numerator

APdocument = numerator / denominator  #APdocument value for each document
APDocmatrix[:,11] = APdocument

#Find the location of any NAN entry and replace with 0.0
nanLoc = np.argwhere(np.isnan(APdocument))
for i in range(len(nanLoc)):
    print("nan@: " + str(nanLoc[i][0]))
    APdocument[nanLoc[i][0]] = 0.0   
    
#Mean of APdocument 
apDocumentMean = np.mean(APdocument)
print("Mean APdocument : " + str(apDocumentMean))

#Variance of APdocument 
APdocumentnVariance = np.var(APdocument)#Variance of APemotion
print("Variance APemotion:" + str(APdocumentnVariance))

###########  Evaluation 4: APemotion ############

A = predict_test # Predicted labels, Aj
B = y_test # Original Labels, Bj

AMean = np.mean(A,axis=0) #Acap
AMean = np.reshape(AMean,(1,5)) #Acap reshaped into 1x5
AMean4docs = np.repeat(AMean, repeats = [len(A)], axis=0) # Repeat AMean vector for all documents

BMean = np.mean(B,axis=0) #Bcap
BMean = np.reshape(BMean,(1,5)) #Bcap reshaped into 1x5
BMean4docs = np.repeat(BMean, repeats = [len(B)], axis=0) # Repeat BMean vector for all documents

AminusAmean = A - AMean4docs  # Aj - Acap
BminusBmean = B - BMean4docs  #Bj - Bcap

AjxBj =  AminusAmean * BminusBmean  #(Aj - Acap) * (Bj - Bcap)
nominator = np.sum(AjxBj, axis = 0) #suummation of ((Aj - Acap) * (Bj - Bcap)) -- > #Nominator 
nominator = np.reshape(nominator,(1,5)) #nominator reshaped into 1x5

docLen = len(A) #document length

sigmaA = np.std(A, axis= 0)  #standerd Deviation of A
sigmaA = np.reshape(sigmaA,(1,5)) #sigmaA reshaped into 1x5

sigmaB = np.std(B, axis= 0) #standerd Deviation of B
sigmaB = np.reshape(sigmaB,(1,5)) #sigmaB reshaped into 1x5

denomi = (docLen - 1) *(sigmaA) * (sigmaB) #Denominator 

APemotion = nominator / denomi  #APemotion value for each document

APemotionMean = np.mean(APemotion) #Mean of APemotion
print("Mean APemotion:" + str(APemotionMean))

APemotionVariance = np.var(APemotion)#Variance of APemotion
print("Variance APemotion:" + str(APemotionVariance))
print("\n")
#correlation coefficient over each  emotion label 
#Labels: Anger → Angry	Fear → Afraid	Joy → Happy	Sadness → Sad	Surprise → Inspired
print("Anger:"+str(APemotion[0][0]))
print("Fear:"+str(APemotion[0][1]))
print("Joy:"+str(APemotion[0][2]))
print("Sadness:"+str(APemotion[0][3]))
print("Surprise:"+str(APemotion[0][4])+ "\n")

###########  Evaluation 1.1: wasserstein_distance ############
wasserDistance_test = 0
wasserDistance_test_alldocs = np.zeros((len(predict_test),1))
for i in range(len(predict_test)):
    wasserDistance_test_alldocs[i] = wasserstein_distance(predict_test[i],y_test[i])
wasserDistance_test = np.mean(wasserDistance_test_alldocs)
print("wasserstein_distance_test = "+ str(wasserDistance_test))