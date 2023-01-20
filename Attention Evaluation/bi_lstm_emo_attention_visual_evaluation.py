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
    print("nan@: " + str(nanLoc[i][0])) #
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

modelVis = Model(inputs=sentence_indices, outputs=fc)

modelVis.summary()
opt = keras.optimizers.Adam(lr=0.0005)
modelVis.compile(loss='mse', optimizer=opt, metrics=['mse'])
modelVis.load_weights('model.h5')
#pred = modelVis.predict(X_te,verbose=1)

###########  Visualization & Evaluation  ############        

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)


def visualize_attention(Sid):
    # Make new model for output predictions and attentions
    '''
    model.get_layer('attention_vec').output:
    attention_vec (Attention)    [(None, 600), (None, 47)] <- We want (None,47) that is the word att
    '''
    X_te = x_train_padded #x_train_padded #x_test_padded #Assign Train or Test to visualize 

    model_att = Model(inputs=modelVis.input, \
                            outputs=[modelVis.output, modelVis.get_layer('attention_vec').output[-1]])
    #idx = np.random.randint(low = 0, high=X_te.shape[0]) # Get a random test
    idx = Sid
    print("News Snippet ID:" + str(idx))

    print("Pre-processed News Snippet:\n" + x_train[idx])
    print("Original Emotion Label:"+ str(y_train[idx]))
    
    print("Original News Snippet:\n" + x_train_original[idx])
    print("Original Emotion Label:"+ str(y_train_original[idx]))

    tokenized_sample = np.trim_zeros(X_te[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict(X_te[idx:idx+1]) # Perform the prediction
    print("Predicted Emotion Label:"+ str(label_probs))

    # Get decoded text and labels
    id2word = dict(map(reversed, tokenizer.word_index.items()))
    decoded_text = [id2word[word] for word in tokenized_sample] 

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    
    attentions_text = attentions[0,0:len(tokenized_sample)] 
    #attentions_text = attentions[0,-len(tokenized_sample):] #for left padding of zeros; eg:[0 0 0 2 3 4 5]
    attentions_text = (attentions_text - np.min(attentions_text)) / (np.max(attentions_text) - np.min(attentions_text))
    for token, attention_score in zip(decoded_text, attentions_text):
        #print(token, attention_score)
        token_attention_dic[token] = attention_score
        

    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Attention output of Preprocessed News Snippet:  </b>"
    for token, attention in token_attention_dic.items():
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),token)  
   
    # Display text enriched with attention scores 
    display(HTML(html_text))
    emotions = ['Anger','Fear','Joy','Sadness','Surprise'] 
    scores = [label_probs[0][0],label_probs[0][1], label_probs[0][2], label_probs[0][3],label_probs[0][4]]
    plt.figure(figsize=(5,2))
    plt.bar(np.arange(len(emotions)), scores, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan'])#, "purple"
    plt.xticks(np.arange(len(emotions)), emotions)
    plt.ylabel('Scores')
    plt.show()

for _ in range(1):
    visualize_attention(50)