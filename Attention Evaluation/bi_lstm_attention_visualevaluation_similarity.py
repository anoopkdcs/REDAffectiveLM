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
max_features = 39088 #27749 #120305 #13260  how many unique words to use (i.e num rows in embedding vector)
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

checkpointer = ModelCheckpoint(filepath='Model.h5', verbose=1, save_best_only=True)
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
best_model = load_model('Model.h5', custom_objects={'Attention': Attention})
predict_test = best_model.predict(x_test_padded)
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

del best_model
best_model = load_model('Model.h5', custom_objects={'Attention': Attention})
predict_val = best_model.predict(x_val_padded)
rms_val = sqrt(mean_squared_error(y_val, predict_val))
print("RMSE_val = "+ str(rms_val))
 
del best_model
best_model = load_model('Model.h5', custom_objects={'Attention': Attention})
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
modelVis.load_weights('Model.h5')
#pred = modelVis.predict(X_te,verbose=1)

###########  Emolex Reader  ############   
import csv
#path emolex = emoWordnet.csv
#path depech mood = DepecheMood_english_token_full.tsv
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


X_te = x_test_padded #x_train_padded #x_test_padded #Assign Train or Test to visualize 
original_docs = x_test_original
original_docs_labels = y_test_original
pre_processed_docs = x_test

Att_eval = np.zeros((len(X_te),12))

Att_eval_EWA = np.array([])
Att_eval_NEA = np.array([])
Att_eval_ENEA = np.array([])

Att_eval_EWCS = np.array([])
Att_eval_NWCS = np.array([])
Att_eval_ENECS = np.array([])

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

  ############## Method #1: Attention Pattern Probability ##################
  #-------------------------------------------------------------------------#
  # Method #1.1: Emotion Word Attention --> EWAd
  
  Aw_vecReal = np.reshape(attentions_text,(1,len(attentions_text)))
  Aw_vec = Aw_vecReal
  Aw_vec[Aw_vec > 0] = 1
  Aw_vec = np.int64(Aw_vec)
  
  Ew_vec =  np.int64(np.zeros((1,len(attentions_text))))
  att_words = decoded_text
  for ewi in range(len(att_words)):
    if att_words[ewi] in emoWordList:
        Ew_vec[0][ewi] = 1
   
  EWA_numerator = np.sum(np.int64(np.logical_and(Aw_vec,Ew_vec)))
  sum_Ew_vec = np.sum(Ew_vec)
  
  if sum_Ew_vec == 0:
    EWA_denominator = 1
  else:
    EWA_denominator = sum_Ew_vec
  
  EWAd = EWA_numerator / EWA_denominator
  Att_eval[docs][0] = EWAd

  if sum_Ew_vec != 0:
    Att_eval_EWA = np.append(Att_eval_EWA,EWAd)
  
  # --------------------------------------------------#
  # Method #1.2 : Named Entity Attention --> NEAd
  Nw_vec = np.int64(np.zeros((1,len(decoded_text))))
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
        Nw_vec[0][loc] = 1
  
  NEA_numerator = np.sum(np.int64(np.logical_and(Aw_vec,Nw_vec)))
  sum_Nw_vec = np.sum(Nw_vec)
  if sum_Nw_vec == 0:
    NEA_denominator = 1
  else:
    NEA_denominator = sum_Nw_vec
  
  NEAd = NEA_numerator / NEA_denominator
  Att_eval[docs][1] = NEAd

  if sum_Nw_vec != 0:
    Att_eval_NEA = np.append(Att_eval_NEA, NEAd)
    

  # --------------------------------------------------#
  #Method #1.3: Emotion word and Named Entity Attention
  ENEw_vec = np.int64(np.logical_or(Ew_vec,Nw_vec)) 
  ENEA_numerator = np.sum(np.int64(np.logical_and(ENEw_vec,Aw_vec)))
  sum_ENEw_vec = np.sum(ENEw_vec)
  
  if sum_ENEw_vec == 0:
    ENEA_denominator = 1
  else:
    ENEA_denominator = sum_ENEw_vec
  
  ENEAd = ENEA_numerator / ENEA_denominator
  Att_eval[docs][2] = ENEAd

  if sum_ENEw_vec != 0:
    Att_eval_ENEA = np.append(Att_eval_ENEA,ENEAd )
  
  
  
  ############ Method #4: Attention Score Similarity Matching  ##########
  # Cosine Similarity Coefficient
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
  # --------------------------------------------------#
  #Method #4.1: Emotion Word Score Similarity in Attention
  Attn_vec = copy.copy(Aw_vec)
  Emo_vec = copy.copy(Ew_vec)

  norm_Attn_vec = Attn_vec / np.sum(Attn_vec)
  EWCS_Emo_vec_sum = np.sum(Emo_vec)
  if EWCS_Emo_vec_sum == 0:
    EWCS_denom = 1
  else:
    EWCS_denom = EWCS_Emo_vec_sum

  norm_Emo_vec = Emo_vec / EWCS_denom
  EWCS_d = cosine_similarity(norm_Attn_vec, norm_Emo_vec)
  Att_eval[docs][9] = EWCS_d

  if EWCS_Emo_vec_sum != 0:
    Att_eval_EWCS = np.append(Att_eval_EWCS,EWCS_d)
  
  # --------------------------------------------------#
  #Method #4.2: NER Score Similarity in Attention
  NER_vec = copy.copy(Nw_vec)
  NWCS_NER_vec_sum = np.sum(NER_vec)
  if NWCS_NER_vec_sum == 0:
    EWCS_denom = 1
  else:
    EWCS_denom = NWCS_NER_vec_sum

  norm_NER_vec = NER_vec / EWCS_denom
  NWCS_d = cosine_similarity(norm_Attn_vec, norm_NER_vec)
  Att_eval[docs][10] = NWCS_d
  
  if NWCS_NER_vec_sum != 0:
    Att_eval_NWCS = np.append(Att_eval_NWCS,NWCS_d)

  # --------------------------------------------------#
  #Method #4.3: Emo word NER Score Similarity in Attention
  EmoNER_vec = copy.copy(ENEw_vec)
  ENECS_EmoNER_vec_sum = np.sum(EmoNER_vec)
  if ENECS_EmoNER_vec_sum == 0:
    ENECS_denom = 1
  else:
    ENECS_denom = ENECS_EmoNER_vec_sum

  norm_EmoNER_vec = EmoNER_vec / ENECS_denom
  ENECS_d = cosine_similarity(norm_Attn_vec, norm_EmoNER_vec)
  Att_eval[docs][11] = ENECS_d
  
  if ENECS_EmoNER_vec_sum != 0:
    Att_eval_ENECS = np.append(Att_eval_ENECS,ENECS_d)


Att_eval_EWA_mean = np.mean(Att_eval_EWA)
Att_eval_NEA_mean = np.mean(Att_eval_NEA)
Att_eval_ENEA_mean = np.mean(Att_eval_ENEA)


Att_eval_EWCS_mean = np.mean(Att_eval_EWCS)
Att_eval_NWCS_mean = np.mean(Att_eval_NWCS)
Att_eval_ENECS_mean = np.mean(Att_eval_ENECS)

print("\n mean for valid documents: D-D'")
print("\nEWAP: " + str(Att_eval_EWA_mean))
print("NEAP: " + str(Att_eval_NEA_mean))
print("ENEAP: " + str( Att_eval_ENEA_mean))


print("\nEWCS: " + str(Att_eval_EWCS_mean ))
print("NECS: " + str(Att_eval_NWCS_mean ))
print("ENECS: " + str( Att_eval_ENECS_mean))

print("\n D-D' ")
print("\nlen of EWAP:"+ str(len(Att_eval_EWA)))
print("len of NEAP:"+ str(len(Att_eval_NEA)))
print("len of ENEAP:"+ str(len(Att_eval_ENEA)))

print("\nlen of EWCS:"+ str(len(Att_eval_EWCS)))
print("len of NECS:"+ str(len(Att_eval_NWCS)))
print("len of ENECS:"+ str(len(Att_eval_ENECS)))

#https://realpython.com/python-histograms/
import pandas as pd
import seaborn as sns
import pylab as pl
import matplotlib.pyplot as plt

#plotHistData = copy.copy(Att_eval)
pandaData_emotionHistogram = pd.DataFrame({'EWAP': Att_eval_EWA})
pandaData_emotionHistogram. plot.hist(grid=True, rwidth=.9, color='#00788a', )# bins=30,
pl.xlabel("EWAP")
pl.ylabel("Document Frequency")
pl.legend().remove() 

pandaData_emotionHistogram1 = pd.DataFrame({'NEAP': Att_eval_NEA})
pandaData_emotionHistogram1. plot.hist(grid=True, rwidth=.9, color='#00788a')# bins=30, #607c8e
pl.xlabel("NEAP")
pl.ylabel("Document Frequency")
pl.legend().remove() 

pandaData_emotionHistogram2 = pd.DataFrame({'ENEAP': Att_eval_ENEA})
pandaData_emotionHistogram2. plot.hist(grid=True, rwidth=.9, color='#00788a')# bins=30,#737678
pl.xlabel("ENEAP")
pl.ylabel("Document Frequency")
pl.legend().remove() 
#--------------------------------------

pandaData_emotionHistogram3 = pd.DataFrame({'EWCS': Att_eval_EWCS})
pandaData_emotionHistogram3. plot.hist(grid=True, rwidth=.9, color='#304157', )# bins=30,
pl.xlabel("EWCS")
pl.ylabel("Document Frequency")
pl.legend().remove() 

pandaData_emotionHistogram4 = pd.DataFrame({'NECS': Att_eval_NWCS})
pandaData_emotionHistogram4. plot.hist(grid=True, rwidth=.9, color='#304157')# bins=30, #607c8e
pl.xlabel("NECS")
pl.ylabel("Document Frequency")
pl.legend().remove() 

pandaData_emotionHistogram5 = pd.DataFrame({'ENECS': Att_eval_ENECS})
pandaData_emotionHistogram5. plot.hist(grid=True, rwidth=.9, color='#304157')# bins=30,#737678
pl.xlabel("ENECS")
pl.ylabel("Document Frequency")
pl.legend().remove() 

plt.show()

import matplotlib.pyplot as plt
x = [Att_eval_EWA, Att_eval_NEA, Att_eval_ENEA,
     Att_eval_EWCS, Att_eval_NWCS, Att_eval_ENECS]      
plt.boxplot(x)
plt.xticks([1, 2, 3, 4, 5, 6], ['A', 'B', 'C', 'D', 'E', 'F'])
#plt.figure(figsize=(40,20))
plt.show()