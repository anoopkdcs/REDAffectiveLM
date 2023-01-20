# -*- coding: utf-8 -*-
"""

"""


# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

import keras
import tensorflow
print(keras.__version__)
print(tensorflow.__version__)

tensorflow.test.gpu_device_name() 
print(tensorflow.test.is_gpu_available())

import numpy as np
import csv
from collections import Counter

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


from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.layers.core import Dropout
from keras.models import load_model
from keras.regularizers import l2
from scipy.stats import wasserstein_distance

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

print("headline_0:- "+str(headlines[0]))
print("label_0:- "+str(labels[0]))
print("detokenized_0:- "+str(deTokenized[0]))
print("detokenized_label_0:- "+str(labels[0]))
print("\n")
print("headline_150:- "+str(headlines[150]))
print("label_150:- "+str(labels[150]))
print("detokenized_150:- "+str(deTokenized[150]))
print("detokenized_label_150:- "+str(labels[150]))

################################ Lexicon Creation ######################
#read DepecheMood
#### Tocken AFRAID	AMUSED	ANGRY	ANNOYED	DONT_CARE	HAPPY	INSPIRED  	SAD	   freq
#### Mapping: Fear → Afraid, Anger → Angry, Joy → Happy, Sadness → Sad and Surprise → Inspired. 

Path_intensityLex= "file.tsv"
file = open(Path_intensityLex,'r')
reader = csv.reader(file, delimiter='\t')
# row[0] = tocken
# anger = row[3]
# fear = row[1]
# joy = row[6]
# sadness = row[8]
# surprise = row[7]

EWNEmolex = []
DepchEmolexDic = {}
for csvrow in reader:
    #print(row[0])
    tempEWNEmolex = [csvrow[0], csvrow[3], csvrow[1],csvrow[6], csvrow[8], csvrow[7]]
    EWNEmolex.append(tempEWNEmolex)
    key = csvrow[0]
    val=[csvrow[3], csvrow[1],csvrow[6], csvrow[8], csvrow[7]]
    tempEWNEmolexDic = {key:val}
    DepchEmolexDic.update(tempEWNEmolexDic)

DepchEmolexDic['aig'][3] = 0 # to remove error value '+AC0-6.77475302732397E+AC0ALQ-005+AC0-' for the word 'aig'
DepchEmolexDic['lpu'][0] = 0 # to remove error value +AC0-5.22881717394546E+AC0ALQ-005+AC0- for the word 'lpu'
DepchEmolexDic['cariaso'] [0] = 0 # to remove error value '+AC0-4.65351830666711E+AC0ALQ-005+AC0-' for the word 'cariaso'
DepchEmolexDic['peaches'] [0] = 0 # to remove error value '+AC0-6.15891691807668E+AC0ALQ-005+AC0-' for the word 'peaches'

################################ Feature Vector Creation ######################
# 1. Document emotion word count
# anger = row[3]
# fear = row[1]
# joy = row[6]
# sadness = row[8]
# surprise = row[7]
dataDim = len(tokenised)
emoDIm = 5 #anger, fear, joy, sadness, surprise
Total_Emotion_Count = np.zeros((dataDim,emoDIm))

'''
Test with muliple word occurance
t = ['cancer']
tokenised[0] =tokenised[0] +t 
'''

for dc in range(len(tokenised)):
    wordCounts = Counter(tokenised[dc])   
    for wrd in wordCounts:
        tmpWord = wrd
        if tmpWord in DepchEmolexDic:
            tempEmovector = DepchEmolexDic[tmpWord]
            tempflotEmovector = list(np.float_(tempEmovector))
            maxIndex = tempflotEmovector.index(max(tempflotEmovector))
            tmpWordcount = wordCounts[tmpWord]
            updations = 1*tmpWordcount
            Total_Emotion_Count [dc][maxIndex] = Total_Emotion_Count [dc][maxIndex] + updations

################################ train test val split ######################
x_train_val, x_test, y_train_val, y_test = train_test_split(Total_Emotion_Count, labels, test_size=0.20, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)
print("x_train: "+str(len(x_train)))
print("y_train: "+str(y_train.shape))

print("x_val: "+str(len(x_val)))
print("y_val: "+str(y_val.shape))

print("x_test: "+str(len(x_test)))
print("y_test: "+str(y_test.shape))

################################ ML models ######################
input_dimen = x_train.shape[1]
model = Sequential()
model.add(Dense(128, input_dim=input_dimen, activation='relu', kernel_regularizer=l2(0.1)))
model.add(Dense(5, activation='softmax'))
epoc = 100
opt = keras.optimizers.Adam(lr=0.0005)#,decay=0.000001
model.compile(loss='mse', optimizer=opt , metrics=['mse'])
checkpointer = ModelCheckpoint(filepath='file.h5', verbose=1, save_best_only=True)
history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=epoc, batch_size=64, callbacks=[checkpointer], verbose=1)
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
del model
best_model = load_model('file.h5')
predict_test = best_model.predict(x_test)
rms_test = sqrt(mean_squared_error(y_test, predict_test))
print("RMSE_test = "+ str(rms_test))

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