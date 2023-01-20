# -*- coding: utf-8 -*-
"""

"""

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

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor

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

################################ train test val split ######################
x_train_val, x_test, y_train_val, y_test = train_test_split(deTokenized, labels, test_size=0.20, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)
print("x_train: "+str(len(x_train)))
print("y_train: "+str(y_train.shape))

print("x_val: "+str(len(x_val)))
print("y_val: "+str(y_val.shape))

print("x_test: "+str(len(x_test)))
print("y_test: "+str(y_test.shape))

################################ N-Grams Creation ######################
vectorizer = TfidfVectorizer(ngram_range=(1,1), min_df=2, stop_words='english')
x_train_vec = vectorizer.fit_transform(x_train)

featureNameTrin = vectorizer.get_feature_names()
NgramTrainFeatures = x_train_vec.toarray()


x_test_vec = vectorizer.transform(x_test)
NgramTestFeatures = x_test_vec.toarray()

################################ ML models ######################
predi1 = MultiOutputRegressor(svm.SVR(kernel='linear')).fit(x_train_vec, y_train).predict(x_test_vec)
rms1= sqrt(mean_squared_error(y_test, predi1))

predi2 = MultiOutputRegressor(linear_model.Ridge()).fit(x_train_vec, y_train).predict(x_test_vec)
rms2= sqrt(mean_squared_error(y_test, predi2))

predi3 = MultiOutputRegressor(GradientBoostingRegressor()).fit(x_train_vec, y_train).predict(x_test_vec)
rms3= sqrt(mean_squared_error(y_test, predi3))

print("RMSE SVR: " + str(rms1))
print("RMSE Ridge: " + str(rms2))
print("RMSE GBR: " + str(rms3))

###########  Evaluation2: Acc@N, N == 1 ############
#Acc@N : N==1
maxEmoPredict_test_svr = np.argmax(predi1,1)
sortdMaxEmoActual_test_svr = np.argsort(-y_test, axis=1)

#Acc@1 SVR
sumAT1_test_svr = np.sum(maxEmoPredict_test_svr == sortdMaxEmoActual_test_svr[:,0]) 
accAT1_test_svr = sumAT1_test_svr / np.size(y_test,0)
print("Acc@1_SVR = "+ str(accAT1_test_svr))


#Acc@N : N==1, Ridge
maxEmoPredict_test_ridge = np.argmax(predi2,1)
sortdMaxEmoActual_test_ridge = np.argsort(-y_test, axis=1)

#Acc@1 Ridge
sumAT1_test_ridge= np.sum(maxEmoPredict_test_ridge == sortdMaxEmoActual_test_ridge[:,0]) 
accAT1_test_ridge = sumAT1_test_ridge / np.size(y_test,0)
print("Acc@1_Ridg = "+ str(accAT1_test_ridge))


#Acc@N : N==1, GBR
maxEmoPredict_test_gbr = np.argmax(predi3,1)
sortdMaxEmoActual_test_gbr = np.argsort(-y_test, axis=1)

#Acc@1 GBR
sumAT1_test_gbr= np.sum(maxEmoPredict_test_gbr == sortdMaxEmoActual_test_gbr[:,0]) 
accAT1_test_gbr = sumAT1_test_gbr / np.size(y_test,0)
print("Acc@1_GBR = "+ str(accAT1_test_gbr))

###########  Evaluation 3: APdocument  ############
X = predi1 # Predicted Labels
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
print("Mean APdocument SVR: " + str(apDocumentMean))

#Variance of APdocument 
APdocumentnVariance = np.var(APdocument)#Variance of APemotion
print("Variance APemotion SVR: " + str(APdocumentnVariance))

###########  Evaluation 3: APdocument  ############
X = predi2 # Predicted Labels
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
print("Mean APdocument Ridge: " + str(apDocumentMean))

#Variance of APdocument 
APdocumentnVariance = np.var(APdocument)#Variance of APemotion
print("Variance APemotion Ridge: " + str(APdocumentnVariance))

###########  Evaluation 3: APdocument  ############
X = predi3 # Predicted Labels
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
print("Mean APdocument GBR: " + str(apDocumentMean))

#Variance of APdocument 
APdocumentnVariance = np.var(APdocument)#Variance of APemotion
print("Variance APemotion GBR: " + str(APdocumentnVariance))

###########  Evaluation 4: APemotion ############

A = predi1 # Predicted labels, Aj
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
print("Mean APemotion SVR:" + str(APemotionMean))

APemotionVariance = np.var(APemotion)#Variance of APemotion
print("Variance APemotion SVR:" + str(APemotionVariance))
print("\n")
#correlation coefficient over each  emotion label 
#Labels: Anger → Angry	Fear → Afraid	Joy → Happy	Sadness → Sad	Surprise → Inspired
print("Anger:"+str(APemotion[0][0]))
print("Fear:"+str(APemotion[0][1]))
print("Joy:"+str(APemotion[0][2]))
print("Sadness:"+str(APemotion[0][3]))
print("Surprise:"+str(APemotion[0][4])+ "\n")

###########  Evaluation 4: APemotion ############

A = predi2 # Predicted labels, Aj
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
print("Mean APemotion Ridge:" + str(APemotionMean))

APemotionVariance = np.var(APemotion)#Variance of APemotion
print("Variance APemotion Ridge:" + str(APemotionVariance))
print("\n")
#correlation coefficient over each  emotion label 
#Labels: Anger → Angry	Fear → Afraid	Joy → Happy	Sadness → Sad	Surprise → Inspired
print("Anger:"+str(APemotion[0][0]))
print("Fear:"+str(APemotion[0][1]))
print("Joy:"+str(APemotion[0][2]))
print("Sadness:"+str(APemotion[0][3]))
print("Surprise:"+str(APemotion[0][4])+ "\n")

###########  Evaluation 4: APemotion ############

A = predi3 # Predicted labels, Aj
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
print("Mean APemotion GBR:" + str(APemotionMean))

APemotionVariance = np.var(APemotion)#Variance of APemotion
print("Variance APemotion GBR:" + str(APemotionVariance))
print("\n")
#correlation coefficient over each  emotion label 
#Labels: Anger → Angry	Fear → Afraid	Joy → Happy	Sadness → Sad	Surprise → Inspired
print("Anger:"+str(APemotion[0][0]))
print("Fear:"+str(APemotion[0][1]))
print("Joy:"+str(APemotion[0][2]))
print("Sadness:"+str(APemotion[0][3]))
print("Surprise:"+str(APemotion[0][4])+ "\n")

###########  Evaluation 1.1: wasserstein_distance ############
wasserDistance_svr = 0
wasserDistance_svr_alldocs = np.zeros((len(predi1),1))
for i in range(len(predi1)):
    wasserDistance_svr_alldocs[i] = wasserstein_distance(predi1[i],y_train[i])
wasserDistance_svr = np.mean(wasserDistance_svr_alldocs)
print("wasserstein distance svr = "+ str(wasserDistance_svr))


wasserDistance_ridge = 0
wasserDistance_ridge_alldocs = np.zeros((len(predi2),1))
for i in range(len(predi2)):
    wasserDistance_ridge_alldocs[i] = wasserstein_distance(predi2[i],y_train[i])
wasserDistance_ridge = np.mean(wasserDistance_ridge_alldocs)
print("wasserstein distance ridge = "+ str(wasserDistance_ridge))

wasserDistance_GBR = 0
wasserDistance_GBR_alldocs = np.zeros((len(predi3),1))
for i in range(len(predi3)):
    wasserDistance_GBR_alldocs[i] = wasserstein_distance(predi3[i],y_train[i])
wasserDistance_GBR = np.mean(wasserDistance_GBR_alldocs)
print("wasserstein distance GBR = "+ str(wasserDistance_GBR))