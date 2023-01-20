# -*- coding: utf-8 -*-
"""
"""


import tensorflow as tf
print(tf.__version__)
#

import keras as k
print(k.__version__)


from tensorflow.keras.layers import Layer

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import transformers

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import string

import matplotlib.pyplot as plt
import re

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

plt.style.use('seaborn')

from transformers import TFXLNetModel, XLNetTokenizer

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import wasserstein_distance


from sklearn.model_selection import train_test_split
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
#from keras import layers
#from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import Concatenate, concatenate
from keras.callbacks import Callback
from keras.regularizers import l2

# Visualization
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
sns.set()
from keras.models import load_model
import os
from scipy.stats import wasserstein_distance

print(tf.config.list_physical_devices('GPU'))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


######################## read data ################################
headlines = np.load('path/headline_abstract-data.npy')
labels = np.load('path/headline_abstract-labels.npy')

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
    tempTokens = tempTokens.translate(str.maketrans('','',"~!@#$%^&*()_-+={}[]|\/><'?.,-+`:;1234567890"))
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

print(x_train[1])
print(x_test[0])
print(x_val[0])

def get_inputs(tweets, tokenizer, max_len=35):
    """ Gets tensors from text using the tokenizer provided"""
    inps = [tokenizer.encode_plus(t, max_length=max_len, pad_to_max_length=True, add_special_tokens=True) for t in tweets]
    inp_tok = np.array([a['input_ids'] for a in inps])
    ids = np.array([a['attention_mask'] for a in inps])
    segments = np.array([a['token_type_ids'] for a in inps])
    return inp_tok, ids, segments

def warmup(epoch, lr):
    """Used for increasing the learning rate slowly, this tends to achieve better convergence.
    However, as we are finetuning for few epoch it's not crucial.
    """
    return max(lr +1e-6, 2e-5)

# This is the identifier of the model. The library need this ID to download the weights and initialize the architecture
# here is all the supported ones:
# https://huggingface.co/transformers/pretrained_models.html
xlnet_model = 'xlnet-large-cased' #xlnet-base-cased-spiece.model, xlnet-large-cased
xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)

x_train_tokens, x_train_ids, x_train_segments = get_inputs(x_train, xlnet_tokenizer)
x_val_tokens, x_val_ids, x_val_segments = get_inputs(x_val, xlnet_tokenizer)
x_test_tokens, x_test_ids, x_test_segments = get_inputs(x_test, xlnet_tokenizer)

print("train token shape:" +str(x_train_tokens.shape))
print("val token shape:" +str(x_val_tokens.shape))
print("test token shape:" +str(x_test_tokens.shape))

print(x_train_tokens[1])
print(x_val_tokens[0])
print(x_test_tokens[0])

###########  Parameters Splitings for Bi-LSTM Model  ############
#EMBEDDING_DIM=100 # how big is each word vector
max_features = 3286 #13260  how many unique words to use (i.e num rows in embedding vector)
maxlen = 10 # max number of words in a comment to use

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(deTokenized)
sequences_train = tokenizer.texts_to_sequences(x_train)
sequences_val = tokenizer.texts_to_sequences(x_val)
sequences_test=tokenizer.texts_to_sequences(x_test)

x_train_padded = pad_sequences(sequences_train,padding='post', maxlen=maxlen)
x_val_padded = pad_sequences(sequences_val,padding='post', maxlen=maxlen)
x_test_padded = pad_sequences(sequences_test,padding='post', maxlen=maxlen)
X_te = x_test_padded
word_index = tokenizer.word_index

##########  Emotion Glove EMbedding ############
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

initializer_one = tensorflow.keras.initializers.Ones()

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
        print("anoop")
        shape1=input_shape
        print(shape1)

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 #initializer=self.init,
                                 initializer=initializer_one,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint,
                                 trainable=False)
        print("anoop2")
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
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
        print("anoop3")
        

        if self.bias:
            print("anoop4")
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
        print("ANoop")

        if self.return_attention:
            return [result, a]
            print(result)
        
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]

def create_xlnet(mname):
    """ Creates the model. It is composed of the XLNet main block and then
    a classification head its added
    """
    # Define token ids as inputs
    word_inputs = tf.keras.Input(shape=(35,), name='word_inputs', dtype='int32')

    #Bi-LSTM model input 
    sentence_indices_glove = Input(name='sentence_indices_glove',shape=[maxlen], dtype='int32')
    embeddings_glove = Embedding(vocabulary_size, embedding_dim, weights=[embedding_matrix],trainable = False)(sentence_indices_glove)

    # Call XLNet model
    xlnet = TFXLNetModel.from_pretrained(mname)
    xlnet_encodings = xlnet(word_inputs)[0]

    #LSTM model
    X_glove = Bidirectional( LSTM(100,return_sequences=True, activation='relu', kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001),))(embeddings_glove)
    X_glove = Dropout(0.5)(X_glove) 
    sentence, word_scores = Attention(return_attention=True, name = "attention_vec")(X_glove)

    # CLASSIFICATION HEAD 
    # Collect last step from last hidden state (CLS)
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    # Apply dropout for regularization
    doc_encoding = tf.keras.layers.Dropout(0.1)(doc_encoding)

    #Concatinating models 
    finalModel = keras.layers.concatenate([doc_encoding, sentence ])
    
    #another fully connected layer
    finalModel = tf.keras.layers.Dense(1224, activation='relu', name='fully_connected_1')(finalModel)

    #another fully connected layer
    finalModel = tf.keras.layers.Dense(1224, activation='relu', name='fully_connected_2')(finalModel)

    #another fully connected layer
    #finalModel = tf.keras.layers.Dense(1224, activation='relu', name='fully_connected_3')(finalModel)
    
    # Final output 
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(finalModel)

    # Compile model
    model = Model(inputs=[word_inputs, sentence_indices_glove], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.000015), loss='mse', metrics=['mse'])

    return model

xlnet = create_xlnet(xlnet_model)

xlnet.summary()

from keras.utils.vis_utils import plot_model
plot_model(xlnet, to_file='file name.png', show_shapes=True, show_layer_names=True)

checkpointer = [tf.keras.callbacks.ModelCheckpoint(filepath='file.h5', verbose=1, save_best_only=True, save_weights_only=True),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.02, restore_best_weights=True),
                #tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0)
                ]

print(x_train_tokens.shape)
print(x_train_padded.shape)
print(y_train.shape)
print("------")
print(x_val_tokens.shape)
print(x_val_padded.shape)
print(y_val.shape)

history = xlnet.fit(x=[x_train_tokens,x_train_padded], y=y_train, validation_data=([x_val_tokens,x_val_padded],y_val), epochs=50, batch_size=64, callbacks=checkpointer)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

best_model = create_xlnet(xlnet_model)
best_model.load_weights('file.h5')

####################### Evaluation1: RMSE ######################
predict_test = best_model.predict([x_test_tokens,x_test_padded])
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

predict_val = best_model.predict([x_val_tokens, x_val_padded])
rms_val = sqrt(mean_squared_error(y_val, predict_val))
print("RMSE_val = "+ str(rms_val))

predict_train = best_model.predict([x_train_tokens, x_train_padded])
rms_train = sqrt(mean_squared_error(y_train, predict_train))
print("RMSE_train = "+ str(rms_train))

predict_test[0:5]

###########  Evaluation 1.1: wasserstein_distance ############
wasserDistance_train = 0
wasserDistance_train_alldocs = np.zeros((len(predict_train),1))
for i in range(len(predict_train)):
    wasserDistance_train_alldocs[i] = wasserstein_distance(predict_train[i],y_train[i])
wasserDistance_train = np.mean(wasserDistance_train_alldocs)
print("wasserstein_distance_train = "+ str(wasserDistance_train))


wasserDistance_val= 0
wasserDistance_val_alldocs = np.zeros((len(predict_val),1))
for i in range(len(predict_val)):
    wasserDistance_val_alldocs[i] = wasserstein_distance(predict_val[i],y_val[i])
wasserDistance_val = np.mean(wasserDistance_val_alldocs)
print("wasserstein_distance_val= "+ str(wasserDistance_val))


wasserDistance_test = 0
wasserDistance_test_alldocs = np.zeros((len(predict_test),1))
for i in range(len(predict_test)):
    wasserDistance_test_alldocs[i] = wasserstein_distance(predict_test[i],y_test[i])
wasserDistance_test = np.mean(wasserDistance_test_alldocs)
print("wasserstein_distance_test = "+ str(wasserDistance_test))

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