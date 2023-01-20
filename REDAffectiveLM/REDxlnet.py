# -*- coding: utf-8 -*-
"""
"""


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

import re


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

plt.style.use('seaborn')

from transformers import TFXLNetModel, XLNetTokenizer

from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import wasserstein_distance

print(tf.__version__)
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

print(x_train[2])

"""
**x_train_tokens** = token id after tokenaization <br>
**x_train_ids** <br>
1. token_ids = 0 indicates List of IDs to which the special tokens will be added. </br>
2. token_ids = 1 (optional) indicates  an Optional second list of IDs for sequence pairs. <br>
**x_train_segments**
<br/>
Segment token indices to indicate first and second portions of the inputs. Indices are selected in 

"""

id = 3
print(x_train_tokens[id])
print(x_train_ids[id])
print(x_train_segments[id])

predicted_token = xlnet_tokenizer.convert_ids_to_tokens(5)
print(predicted_token)

print("train token shape:" +str(x_train_tokens.shape))
print("val token shape:" +str(x_val_tokens.shape))
print("test token shape:" +str(x_test_tokens.shape))

print(x_train_tokens[1])
print(x_val_tokens[0])
print(x_test_tokens[0])

def create_xlnet(mname):
    """ Creates the model. It is composed of the XLNet main block and then
    a classification head its added
    """
    # Define token ids as inputs
    word_inputs = tf.keras.Input(shape=(35,), name='word_inputs', dtype='int32')

    # Call XLNet model
    xlnet = TFXLNetModel.from_pretrained(mname)
    xlnet_encodings = xlnet(word_inputs)[0]

    # CLASSIFICATION HEAD 
    # Collect last step from last hidden state (CLS)
    doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
    # Apply dropout for regularization
    doc_encoding = tf.keras.layers.Dropout(.5)(doc_encoding)
    # Final output 
    outputs = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(doc_encoding)

    # Compile model
    model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0000015), loss='mse', metrics=['mse'])

    return model

xlnet = create_xlnet(xlnet_model)

xlnet.summary()

'''
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=4, min_delta=0.02, restore_best_weights=True),
    tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=1e-6, patience=2, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-6)
]


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True, verbose=1)
lr_callback = LearningRateScheduler(lrfn, verbose=1)
callback_list = [checkpoint, es, lr_callback]
'''

checkpointer = [tf.keras.callbacks.ModelCheckpoint(filepath='file.h5', verbose=1, save_best_only=True, save_weights_only=True),
                #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.02, restore_best_weights=True),
                #tf.keras.callbacks.LearningRateScheduler(warmup, verbose=0)
                ]

history = xlnet.fit(x=x_train_tokens, y=y_train, validation_data=(x_val_tokens,y_val), epochs=1000, batch_size=64,  callbacks=[checkpointer]) #,  callbacks=checkpointer

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
predict_test = best_model.predict(x_test_tokens)
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

predict_val = best_model.predict(x_val_tokens)
rms_val = sqrt(mean_squared_error(y_val, predict_val))
print("RMSE_val = "+ str(rms_val))

predict_train = best_model.predict(x_train_tokens)
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