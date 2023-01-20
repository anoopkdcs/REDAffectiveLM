# -*- coding: utf-8 -*-
"""
"""

import keras
import tensorflow
print(keras.__version__)
print(tensorflow.__version__)

tensorflow.test.gpu_device_name() 
print(tensorflow.test.is_gpu_available())

import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
nltk.download('stopwords')

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

# Visualization
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
sns.set()
from keras.models import load_model
import os

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
    tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;1234567890ï½ãï½âï½â�Ðð"))
    tempTokens = tokenizer.tokenize(tempTokens) #tokenization 
    
#    for j in range(len(tempTokens)):
#        tempTokens[j] = lemmatizer.lemmatize(tempTokens[j] , get_wordnet_pos(tempTokens[j] )) #lemetization
        
    tempTokensStopRemoval = [word for word in tempTokens if word not in stop_words] #stopword removal 
    Tokens.append(tempTokens) # tokens with out stopword removal 
    finalTokens.append(tempTokensStopRemoval) # tokens after stopword removal

# De-tokenized sentences
deTokenized = []
for j in range(len(finalTokens)):
    tempTokens = []
    tempDetoken = finalTokens[j]
    tempDetoken = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tempDetoken]).strip()
    deTokenized.append(tempDetoken)

tokenised =  finalTokens

print("headline_0:- "+str(headlines[0]))
print("label_0:- "+str(labels[0]))
print("detokenized_0:- "+str(deTokenized[0]))
print("detokenized_label_0:- "+str(labels[0]))
print("\n")
print("headline_150:- "+str(headlines[150]))
print("label_150:- "+str(labels[150]))
print("detokenized_150:- "+str(deTokenized[150]))
print("detokenized_label_150:- "+str(labels[150]))

################################ train test val split ######################

x_train_val, x_val, y_train_val, y_val = train_test_split(deTokenized, labels, test_size=0.2, random_state=seed)
x_train, x_test, y_train, y_test  = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)


print("x_train: "+str(len(x_train)))
print("y_train: "+str(y_train.shape))

print("x_val: "+str(len(x_val)))
print("y_val: "+str(y_val.shape))

print("x_test: "+str(len(x_test)))
print("y_test: "+str(y_test.shape))

################################ train test val split Not PREPROCESSED - Used for visualization ######################

x_train_val_original, x_val_original, y_train_val_original, y_val_original = train_test_split(headlines, labels, test_size=0.2, random_state=seed)
x_train_original, x_test_original, y_train_original, y_test_original  = train_test_split(x_train_val_original, y_train_val_original, test_size=0.25, random_state=seed)


print("x_train: "+str(len(x_train_original)))
print("y_train: "+str(y_train_original.shape))

print("x_val: "+str(len(x_val_original)))
print("y_val: "+str(y_val_original.shape))

print("x_test: "+str(len(x_test_original)))
print("y_test: "+str(y_test_original.shape))

print("x_train_0:- "+str(x_train[0]))
print("label_0:- "+str(y_train[0]))
print("x_train_original_0:- "+str(x_train_original[0]))
print("label_original_0:- "+str(y_train_original[0]))
print("\n")
print("x_train_200:- "+str(x_train[200]))
print("label_200:- "+str(y_train[200]))
print("x_train_original_200:- "+str(x_train_original[200]))
print("label_original_200:- "+str(y_train_original[200]))
print("\n")

print("x_test_0:- "+str(x_test[0]))
print("label_0:- "+str(y_test[0]))
print("x_test_original_0:- "+str(x_test_original[0]))
print("label_0:- "+str(y_test_original[0]))
print("\n")
print("x_test_200:- "+str(x_test[200]))
print("label_200:- "+str(y_test[200]))
print("x_test_original_200:- "+str(x_test_original[200]))
print("label_200:- "+str(y_test_original[200]))
print("\n")

print("x_val_0:- "+str(x_val[0]))
print("label_0:- "+str(y_val[0]))
print("x_val_original_0:- "+str(x_val_original[0]))
print("label_original_0:- "+str(y_val_original[0]))
print("\n")
print("x_val_200:- "+str(x_val[200]))
print("label_200:- "+str(y_val[200]))
print("x_train_original_200:- "+str(x_val_original[200]))
print("label_original_200:- "+str(y_val_original[200]))
print("\n")

###########  Parameters Splitings for Bi-LSTM Model  ############
#EMBEDDING_DIM=100 # how big is each word vector
max_features =39088 #27749 #120305 #13260  how many unique words to use (i.e num rows in embedding vector)
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

###########  Attention Layer Defenition  ############
class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, return_attention=False,
                 **kwargs):
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)


    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

##########  Glove EMbedding for Bi-LSTM Model  ############

embeddings_index = {}
f = open(os.path.join('path', 'em-glove.6B.100d_20epochs.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


print("Embedding Strted")
embedding_dim = 100 # how big is each word vector
vocabulary_size = min(max_features, len(word_index)) + 1

# first create a matrix of zeros, this is our embedding matrix
embedding_matrix = np.zeros((vocabulary_size, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)

print("Embedding END")

################## LSTM Model ##############
sentence_indices = Input(name='sentence_indices',shape=[maxlen], dtype='int32') #input_shape -- shape of the input, usually (max_len,)
embeddings = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],trainable = False)(sentence_indices)
X = Bidirectional(LSTM(100, return_sequences=True, 
         activation='relu', 
         kernel_regularizer=l2(0.001), 
         recurrent_regularizer=l2(0.001),
         ))(embeddings) 
X = Dropout(0.5)(X)
sentence, word_scores = Attention(return_attention=True, name = "attention_vec")(X)
fc = Dense(5,activation='softmax')(sentence)
#fc = Activation('softmax')(fc) 

model = Model(inputs=sentence_indices, outputs=fc)

model.summary()
opt = keras.optimizers.Adam(lr=0.0005)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])

checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1, save_best_only=True)
history = model.fit(x_train_padded, y_train, validation_data=(x_val_padded,y_val), epochs=300, batch_size=128, callbacks=[checkpointer], verbose=1)

####################### Plot training & validation loss values ####################
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

####################### Evaluation1: RMSE ######################
del model
best_model = load_model('model.h5', custom_objects={'Attention': Attention})
predict_test = best_model.predict(x_test_padded)
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

del best_model
best_model = load_model('model.h5', custom_objects={'Attention': Attention})
predict_val = best_model.predict(x_val_padded)
rms_val = sqrt(mean_squared_error(y_val, predict_val))
print("RMSE_val = "+ str(rms_val))
 
del best_model
best_model = load_model('model.h5', custom_objects={'Attention': Attention})
predict_train = best_model.predict(x_train_padded)
rms_train = sqrt(mean_squared_error(y_train, predict_train))
print("RMSE_train = "+ str(rms_train))

predict_test[0:5]

###########  Evaluation2: Acc@N, N == 1, 2, 3 ############
#Acc@N : N==1, 2, 3
maxEmoPredict_test = np.argmax(predict_test,1)
sortdMaxEmoActual_test = np.argsort(-y_test, axis=1)

#Acc@1
sumAT1_test = np.sum(maxEmoPredict_test == sortdMaxEmoActual_test[:,0]) 
accAT1_test = sumAT1_test / np.size(y_test,0)
print("Acc@1_test = "+ str(accAT1_test))

###########  Evaluation2: Acc@N, N == 1, 2, 3 ############
#Acc@N : N==1, 2, 3
maxEmoPredict_val = np.argmax(predict_val,1)
sortdMaxEmoActual_val = np.argsort(-y_val, axis=1)
#Acc@1
sumAT1_val = np.sum(maxEmoPredict_val == sortdMaxEmoActual_val[:,0]) 
accAT1_val = sumAT1_val / np.size(y_val,0)
print("Acc@1_val = "+ str(accAT1_val))

###########  Evaluation2: Acc@N, N == 1, 2, 3 ############
#Acc@N : N==1, 2, 3
maxEmoPredict_train = np.argmax(predict_train,1)
sortdMaxEmoActual_train = np.argsort(-y_train, axis=1)
#Acc@1
sumAT1_train = np.sum(maxEmoPredict_train == sortdMaxEmoActual_train[:,0]) 
accAT1_train = sumAT1_train / np.size(y_train,0)
print("Acc@1_train = "+ str(accAT1_train))

################## LSTM Model Reload for Visualization ##############
sentence_indices = Input(name='sentence_indices',shape=[maxlen], dtype='int32') #input_shape -- shape of the input, usually (max_len,)
embeddings = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],trainable = False)(sentence_indices)
X = Bidirectional(LSTM(100, return_sequences=True, 
         activation='relu', 
         kernel_regularizer=l2(0.001), 
         recurrent_regularizer=l2(0.001),
         ))(embeddings) 
X = Dropout(0.5)(X)
sentence, word_scores = Attention(return_attention=True, name = "attention_vec")(X)
fc = Dense(5,activation='softmax')(sentence)
#fc = Activation('softmax')(fc) 

modelVis = Model(inputs=sentence_indices, outputs=fc)

modelVis.summary()
opt = keras.optimizers.Adam(lr=0.0005)
modelVis.compile(loss='mse', optimizer=opt, metrics=['mse'])
modelVis.load_weights('model.h5')
#pred = modelVis.predict(X_te,verbose=1)

###########  Emolex Reader  ############   
import csv
#path emolex = emoWordnet.csv
#path depech mood = DepecheMood_english_token_full.tsv"
Path_intensityLex= "emoWordnet.csv"
file = open(Path_intensityLex,'r')
reader = csv.reader(file, delimiter=',') # depech mood = delimiter='\t' and emo wordnet = delimiter=','
emoWordList = []
for csvrow in reader:
    key = csvrow[0]
    emoWordList.append(key)

#for depech mood len = 187942
#for emo wordnet len = 67738

len(emoWordList) #

###########  Visualization & Evaluation  ############        
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import cosine_similarity
import copy
from sklearn import metrics
import math
import pandas as pd


X_te = x_test_padded #x_train_padded #x_test_padded #Assign Train or Test to visualize 
original_docs = x_test_original
original_docs_labels = y_test_original
pre_processed_docs = x_test

Att_eval = np.zeros((len(X_te),12))

emotion_pair_wise_sim = np.array([])
NER_pair_wise_sim =  np.array([])
emo_NER_pair_wise_sim = np.array([])

Att_eval_NEA = np.array([])
Att_eval_ENEA = np.array([])

nan_count = 0

for docs in range(len(X_te)):#len(X_te)

  print("doc:" +str(docs))
  model_att = Model(inputs=modelVis.input, \
                          outputs=[modelVis.output, modelVis.get_layer('attention_vec').output[-1]])
  idx = docs  
  tokenized_sample = np.trim_zeros(X_te[idx]) # Get the tokenized text
  label_probs, attentions = model_att.predict(X_te[idx:idx+1]) # Perform the prediction
  
  # Get decoded text and labels
  id2word = dict(map(reversed, tokenizer.word_index.items()))
  decoded_text = [id2word[word] for word in tokenized_sample] 
  
  # Get word attentions using attenion vector
  token_attention_dic = {}
  max_score = 0.0
  min_score = 0.0
  
  attentions_text = attentions[0,0:len(tokenized_sample)] 
  #attentions_text = attentions[0,-len(tokenized_sample):] where -len usefull for default padding;means left padding of zeros; eg:[0 0 0 2 3 4 5]
  attentions_text = (attentions_text - np.min(attentions_text)) / (np.max(attentions_text) - np.min(attentions_text))
  attentions_text_original = copy.copy(attentions_text)


  for token, attention_score in zip(decoded_text, attentions_text):
      token_attention_dic[token] = attention_score
  
  #print("decoded_text:"+ str(decoded_text))
  #print("attentions_text:"+ str(attentions_text))
  
  ############## Method #1: Emotion and NER Behaviour Similarity ##################
  #-------------------------------------------------------------------------#
  # Method #1.1: Emotion Similarity
  
  Aw_vecReal = copy.copy(np.reshape(attentions_text,(1,len(attentions_text))))
  emo_MAM = np.zeros(Aw_vecReal.shape)
  #print("attentions vector:"+ str(Aw_vecReal))
  #print(emo_MAM)

  for i in range(len(decoded_text)):
    if (decoded_text[i] in emoWordList) and (decoded_text[i] != 0):
      #print(decoded_text[i])
      emo_MAM[0][i] = Aw_vecReal[0][i]  
  
  EmoAM = np.zeros(Aw_vecReal.shape)
  for j in range(len(decoded_text)):
    if(decoded_text[j] in emoWordList):
      EmoAM[0][j] = 1

  #print(emo_MAM)
  #print(EmoAM)
  #AUC Computation 
  fpr, tpr, thresholds = metrics.roc_curve(EmoAM[0], emo_MAM[0])
  auc = metrics.auc(fpr, tpr)
  #print(auc)
  
  if math.isnan(auc):
    nan_count = nan_count +1
  else:
    #print("emo AUC:" + str(auc))
    emotion_pair_wise_sim = np.append(emotion_pair_wise_sim,auc) 

# --------------------------------------------------#
  # Method #1.2 : Named Entity Behaviour Similarity

  NER_MAM = np.zeros(Aw_vecReal.shape)
  NER_AM = np.int64(np.zeros((1,len(decoded_text))))
  attentiov_vec_real = copy.copy(np.reshape(attentions_text,(1,len(attentions_text))))

  original_doci = np.str(original_docs[idx])
  attention_text = np.array(decoded_text)
  
  doci = nlp(original_doci)
  NER_doc = [(X, X.ent_type_) for X in doci]
  for i in range(len(NER_doc)):
    if NER_doc[i][1] != '':
      NER_word = np.str(NER_doc[i][0])
      NER_lower_word = NER_word.lower()  #np.char.lower(wod)
      NER_symbol_removed = NER_lower_word.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;1234567890"))
      if NER_symbol_removed in attention_text:
        loc = np.where(attention_text == NER_symbol_removed)
        NER_AM[0][loc] = 1
  
  for j in range(len(NER_AM[0])):
    if (NER_AM[0][j] != 0) and  (attentiov_vec_real[0][j] > 0):
      NER_MAM[0][j] = attentiov_vec_real[0][j]
  
  if (np.sum(NER_AM)!=0):
    #AUC Computation 
    fpr, tpr, thresholds = metrics.roc_curve(NER_AM[0], NER_MAM[0])
    auc_NER = metrics.auc(fpr, tpr)
    #print(auc_NER)
    if math.isnan(auc_NER):
      nan_count = nan_count +1
    else:
      #print("NER AUC:" +str(auc_NER))
      NER_pair_wise_sim = np.append(NER_pair_wise_sim,auc_NER)


 # --------------------------------------------------#
  #Method #1.3: Emotion word and Named Entity Behaviour Similarity 
  #if ((np.sum(NER_AM)) == 0) and (pd.notnull(auc)):
    #emo_NER_pair_wise_sim = np.append(emo_NER_pair_wise_sim,auc)
  emo_NER_AM = np.int64(np.logical_or(EmoAM,NER_AM))
  emo_NER_MAM = np.zeros(Aw_vecReal.shape)
  for k in range(len(emo_NER_AM[0])):
    if (emo_NER_AM[0][k] != 0) and (attentiov_vec_real[0][j] > 0):
      emo_NER_MAM[0][k] = attentiov_vec_real[0][k]

  if (np.sum(emo_NER_AM)) != 0:
    #AUC Computation 
    fpr, tpr, thresholds = metrics.roc_curve(emo_NER_AM[0], emo_NER_MAM[0])
    auc_emo_NER = metrics.auc(fpr, tpr)
    #print(auc_NER)
    if math.isnan(auc_emo_NER):
      nan_count = nan_count +1
    else:
      #print("emo NER AUC: "+str(auc_emo_NER))
      emo_NER_pair_wise_sim = np.append(emo_NER_pair_wise_sim,auc_emo_NER)

emo_behaviourSim = np.mean(emotion_pair_wise_sim)
print("emo_behaviourSim: "+str(emo_behaviourSim))

NER_behaviourSim = np.mean(NER_pair_wise_sim)
print("NER_behaviourSim: "+str(NER_behaviourSim))

emo_NER_behaviourSim = np.mean(emo_NER_pair_wise_sim)
print("emo_NER_behaviourSim: "+str(emo_NER_behaviourSim))

print("\n D-D' ")
print("\nlen of emotion_pair_wise_sim:"+ str(len(emotion_pair_wise_sim)))
print("len of NER_pair_wise_sim:"+ str(len(NER_pair_wise_sim)))
print("len of emo_NER_pair_wise_sim:"+ str(len(emo_NER_pair_wise_sim)))