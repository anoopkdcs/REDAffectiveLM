# -*- coding: utf-8 -*-
"""
ETM

"""

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, OrderedDict
from sklearn.metrics import mean_squared_error
from math import sqrt
nltk.download('stopwords')
import string

from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from google.colab import drive
drive.mount('/content/drive')

######################## read data ################################
headlines = np.load('/content/drive/MyDrive/REN20k_short_text/REN-20k_headline_abstract_data.npy')
labels = np.load('/content/drive/MyDrive/REN20k_short_text/REN-20k_headline_abstract_labels.npy')
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
x_train_val, x_test, y_train_val, y_test = train_test_split(tokenised, labels, test_size=0.20, random_state=seed)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=seed)
print("x_train: "+str(len(x_train)))
print("y_train: "+str(y_train.shape))

print("x_val: "+str(len(x_val)))
print("y_val: "+str(y_val.shape))

print("x_test: "+str(len(x_test)))
print("y_test: "+str(y_test.shape))

############################## Training #################
#Gama_de == document vs. {Anger, Fear, Joy, Sadness, Surprise} matrix
gamma_de = y_train

#Delta_dw  == document vs. word count matrix
v = DictVectorizer()
X = v.fit_transform(Counter(f) for f in x_train)
delta_dw = np.int64(X.A)
vocab = v.vocabulary_

# to veryty the cont  use the below code 
d = Counter(delta_dw[1]) 

# Word vs. emotion matrix
wordEmo = np.zeros((len(vocab),5)) #numerator
s = 1

for i in range(len(vocab)):
    for j in range(5):
        wordEmo[i][j] = s +  sum(delta_dw[:,i] * gamma_de[:,j])
        
denominator = np.reshape(np.sum(wordEmo,0), (1,5))

#Probablity of word given emotion matrix
probWordEmo = np.divide(wordEmo,denominator)

############################## Testing #################
#Delta_dw for Testing  == document vs. word count matrix
v = DictVectorizer()
Xtest = v.fit_transform(Counter(f) for f in x_test)
delta_dwTest = Xtest.A
vocabTest = v.vocabulary_

#Prediction using bayes theorem 
docs= x_test 

#prob(e) : priori probablity of emotion e
probE = np.sum(gamma_de,0)/np.size(gamma_de,0)  

#probability of emotion given document
lenDoc = len(docs) 
probEmoDoc = np.zeros((lenDoc,5)) 
for i in range(5):
    for j in range(lenDoc):        
        words = len(docs[j])
        indArray = np.zeros((words,4))
        for k in range(words):
            if docs[j][k] in vocab:
                indArray[k,0] = vocab[docs[j][k]] #word index
                indx_in_vocab = np.int(indArray[k,0])
                indArray[k,1] = probWordEmo[indx_in_vocab][i] #Prob(word given emotion)
                indx_in_vocabtest = vocabTest[docs[j][k]]
                indArray[k,2] = delta_dwTest[j][indx_in_vocabtest] #delta_document,word
                indArray[k,3] = np.power(indArray[k,1],indArray[k,2]) #[Prob(word given emotion)]^[delta_document,word]
            else:
                indArray[k,3] = 1                         
        productTemp = np.product(indArray[:,3]) # product[Prob(word given emotion)]^[delta_document,word]
        probEmoDoc[j][i]  = probE[i] * productTemp #prob(e) * [product[Prob(word given emotion)]^[delta_document,word]]

####################### Evaluation1: RMSE ######################

predict_test = probEmoDoc
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

#if zero in denominator 
loc_zero = np.where(denominator == 0)
loc = np.array(loc_zero)
for r in range(len(loc)):
  for c in range(len(loc[r])):
    ind = loc[r][c]
    if denominator[ind] == 0:
      denominator[ind] = 1


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