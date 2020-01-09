
# coding: utf-8

# In[5]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
import random


# In[6]:


with open('same.csv', 'r') as f:
    reader = csv.reader(f)
    count=0
    t=[]
    cmatrix = []
    smatrix= []
    for row in reader:
        if (count<2000):
            if (count!=0):
                t.append(int(row[2]))
                data=[]
                data1=[]
                with open('GSC-Features.csv','r') as f1:
                    reader1 = csv.reader(f1)
                    count1=0
                    for row1 in reader1:
                        if (count1!=0):
                            if (row1[0]==row[0]):
                                for column in row1:
                                    data.append(column)
                                break
                        count1=count1+1
                    data=data[1:]
                    data=[int(data) for data in data]
                count2=0
                with open('GSC-Features.csv','r') as f1:
                    reader1 = csv.reader(f1)
                    for row1 in reader1:
                        if (count2!=0):
                            if (row1[0]==row[1]):
                                for column in row1:
                                    data1.append(column)
                                break
                        count2=count2+1
                    data1=data1[1:]
                    data1=[int(data1) for data1 in data1]
                    concatenate=data+data1
                    sub=[abs((data)-(data1)) for data,data1 in zip(data,data1)]
                    cmatrix.append(concatenate) 
                    smatrix.append(sub)
        count=count+1   
        


# In[7]:


with open('diffn.csv', 'r') as f:
    reader = csv.reader(f)
    count=0

    
    
    for row in reader:
        if (count <2200):
            if (count!=0):
                t.append(int(row[2]))
                data=[]
                data1=[]
                with open('GSC-Features.csv','r') as f1:
                    reader1 = csv.reader(f1)
                    count1=0
                    for row1 in reader1:
                        if (count1!=0):
                            if (row1[0]==row[0]):
                                for column in row1:
                                    data.append(column)
                                break
                        count1=count1+1
                    data=data[1:] 
                    data=[int(data) for data in data]
                count2=0
                with open('GSC-Features.csv','r') as f1:
                    reader1 = csv.reader(f1)
                    for row1 in reader1:
                        if (count2!=0):
                            if (row1[0]==row[1]):
                                for column in row1:
                                    data1.append(column)
                                break
                        count2=count2+1
                    data1=data1[1:]
                    data1=[int(data1) for data1 in data1]
                    concatenate=data+data1
                    sub=[abs((data)-(data1)) for data,data1 in zip(data,data1)]
                    cmatrix.append(concatenate) 
                    smatrix.append(sub)
            count=count+1    
           


# In[8]:


delc=(np.where(~np.array(cmatrix).any(axis=0))[0])
dels=(np.where(~np.array(smatrix).any(axis=0))[0])
cmatrix = np.delete(cmatrix, delc, axis=1)
smatrix = np.delete(smatrix, dels, axis=1)


# In[9]:


combined = list(zip(smatrix,cmatrix,t))
random.shuffle(combined)
rcmatrix=[]
rsmatrix=[]
rt=[]
rsmatrix[:], rcmatrix[:], rt[:] = zip(*combined)
#print(np.array(rcmatrix).shape)
#print(np.array(rsmatrix).shape)


# In[10]:


#count=0
#for row in rcmatrix:
#    if (row==[2, 1, 1, 0, 2, 2, 0, 1, 2, 1, 0, 1, 3, 2, 2, 0, 4, 2]):
#        break
#    count=count+1
#    
#print (count)


# In[11]:


def GenerateTrainingTarget(rawTraining,TrainingPercent = 80):
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData[0])*0.01*TrainingPercent))
    d2 = rawData[:,0:T_len]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData[0])*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[:,TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t

def GenerateBigSigma(Data, MuMatrix,TrainingPercent,IsSynthetic):
    BigSigma    = np.zeros((len(Data),len(Data)))
    DataT       = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))        
    varVect     = []
    for i in range(0,len(DataT[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[i][j])    
        varVect.append(np.var(vct))
    
    for j in range(len(Data)):
        BigSigma[j][j] = varVect[j]
    if IsSynthetic == True:
        BigSigma = np.dot(3,BigSigma)
    else:
        BigSigma = np.dot(200,BigSigma)
    ##print ("BigSigma Generated..")
    return BigSigma

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetWeightsClosedForm(PHI, T, Lambda):
    Lambda_I = np.identity(len(PHI[0]))
    for i in range(0,len(PHI[0])):
        Lambda_I[i][i] = Lambda
    PHI_T       = np.transpose(PHI)
    PHI_SQR     = np.dot(PHI_T,PHI)
    PHI_SQR_LI  = np.add(Lambda_I,PHI_SQR)
    PHI_SQR_INV = np.linalg.inv(PHI_SQR_LI)
    INTER       = np.dot(PHI_SQR_INV, PHI_T)
    W           = np.dot(INTER, T)
    ##print ("Training Weights Generated..")
    return W

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data)
    TrainingLen = math.ceil(len(DataT)*(TrainingPercent*0.01))         
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) 
    BigSigInv = np.linalg.inv(BigSigma)
    for  C in range(0,len(MuMatrix)):
        for R in range(0,int(TrainingLen)):
            PHI[R][C] = GetRadialBasisOut(DataT[R], MuMatrix[C], BigSigInv)
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)):
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2)
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]):
            counter+=1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT)))
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT))))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# In[12]:


#TrainingTarget = np.array(GenerateTrainingTarget(t,80))
#TrainingC   = GenerateTrainingDataMatrix(cmatrix,80)
#TrainingS = GenerateTrainingDataMatrix(smatrix,80) 
#print(TrainingTarget.shape)
#print(TrainingC.shape)
#print(TrainingS.shape)


# In[13]:


TrainingTarget = np.array(GenerateTrainingTarget(rt,80))
TrainingC   = GenerateTrainingDataMatrix(np.transpose(rcmatrix),80)
TrainingS = GenerateTrainingDataMatrix(np.transpose(rsmatrix),80) 
print("Training Data Shape")
print(TrainingTarget.shape)
print(TrainingC.shape)
print(TrainingS.shape)


# In[14]:


#ValDataAct = np.array(GenerateValTargetVector(t,10, (len(TrainingTarget))))
#ValDataC    = GenerateValData(cmatrix,10, (len(TrainingTarget)))
#ValDataS    = GenerateValData(smatrix,10, (len(TrainingTarget)))
#print(ValDataAct.shape)
#print(ValDataC.shape)
#print(ValDataS.shape)


# In[15]:


ValDataAct = np.array(GenerateValTargetVector(rt,10, (len(TrainingTarget))))
ValDataC    = GenerateValData(np.transpose(rcmatrix),10, (len(TrainingTarget)))
ValDataS    = GenerateValData(np.transpose(rsmatrix),10, (len(TrainingTarget)))
print()
print("Validation Data Shape")

print(ValDataAct.shape)
print(ValDataC.shape)
print(ValDataS.shape)


# In[16]:


#TestDataAct = np.array(GenerateValTargetVector(t,10, (len(TrainingTarget)+len(ValDataAct))))
#TestDataC = GenerateValData(cmatrix,10, (len(TrainingTarget)+len(ValDataAct)))
#TestDataS = GenerateValData(smatrix,10, (len(TrainingTarget)+len(ValDataAct)))
#print(TestDataAct.shape)
#print(TestDataC.shape)
#print(TestDataS.shape)


# In[17]:


TestDataAct = np.array(GenerateValTargetVector(rt,10, (len(TrainingTarget)+len(ValDataAct))))
TestDataC = GenerateValData(np.transpose(rcmatrix),10, (len(TrainingTarget)+len(ValDataAct)))
TestDataS = GenerateValData(np.transpose(rsmatrix),10, (len(TrainingTarget)+len(ValDataAct)))
print()
print("Testing Data Shape")

print(TestDataAct.shape)
print(TestDataC.shape)
print(TestDataS.shape)


# In[18]:


rcmatrix = np.transpose(rcmatrix)     
rsmatrix = np.transpose(rsmatrix)  



# In[19]:


C_Lambda=0.01
M=25
IsSynthetic=False


# In[20]:


kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingC))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(rcmatrix, Mu, 80,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(rcmatrix, Mu, BigSigma, 80)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestDataC, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValDataC, Mu, BigSigma, 100)

TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))
print()
print ("-------Closed Form with Radial Basis Function-------")
print ('----------------Feature Concatenation---------------')
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Training Dataset Accuracy   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Validation Dataset Accuracy = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Testing Dataset Accuracy    = " + str(float(TestAccuracy.split(',')[0])))
print()
W_Now        = np.dot(220, W)
La           = 2 # Regularization Term: added to prevent over-fitting
learningRate = 0.02


for i in range(0,1266):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    # ∇ ED = −(tn −w(τ)'φ(xn))φ(xn)
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    # ∇ E = ∇ ED + λ*∇ EW
    # Stochastic Gradient Descent Solution for w
    
    #-----------------TrainingData Accuracy---------------------#
TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    
    #-----------------ValidationData Accuracy---------------------#
VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    
    #-----------------TestingData Accuracy---------------------#
TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
Erms_Test = GetErms(TEST_OUT,TestDataAct)

print ('----------Gradient Descent Solution-----------------')
print ('----------------Feature Concatenation---------------')
print ("E_rms Training   = " + str(Erms_TR.split(',')[1]))
print ("E_rms Validation = " + str(Erms_Val.split(',')[1]))
print ("E_rms Testing    = " + str(Erms_Test.split(',')[1]))
print ("Training Dataset Accuracy   = " + str(Erms_TR.split(',')[0]))
print ("Validation Dataset Accuracy = " + str(Erms_Val.split(',')[0]))
print ("Testing Dataset Accuracy    = " + str(Erms_Test.split(',')[0]))
print()

# In[22]:


kmeans = KMeans(n_clusters=M, random_state=0).fit(np.transpose(TrainingS))
Mu = kmeans.cluster_centers_

BigSigma     = GenerateBigSigma(rsmatrix, Mu, 80,IsSynthetic)
TRAINING_PHI = GetPhiMatrix(rsmatrix, Mu, BigSigma, 80)
W            = GetWeightsClosedForm(TRAINING_PHI,TrainingTarget,(C_Lambda)) 
TEST_PHI     = GetPhiMatrix(TestDataS, Mu, BigSigma, 100) 
VAL_PHI      = GetPhiMatrix(ValDataS, Mu, BigSigma, 100)

TR_TEST_OUT  = GetValTest(TRAINING_PHI,W)
VAL_TEST_OUT = GetValTest(VAL_PHI,W)
TEST_OUT     = GetValTest(TEST_PHI,W)

TrainingAccuracy   = str(GetErms(TR_TEST_OUT,TrainingTarget))
ValidationAccuracy = str(GetErms(VAL_TEST_OUT,ValDataAct))
TestAccuracy       = str(GetErms(TEST_OUT,TestDataAct))

print ("-------Closed Form with Radial Basis Function-------")
print ('----------------Feature Subtraction---------------')
print ("E_rms Training   = " + str(float(TrainingAccuracy.split(',')[1])))
print ("E_rms Validation = " + str(float(ValidationAccuracy.split(',')[1])))
print ("E_rms Testing    = " + str(float(TestAccuracy.split(',')[1])))
print ("Training Dataset Accuracy   = " + str(float(TrainingAccuracy.split(',')[0])))
print ("Validation Dataset Accuracy = " + str(float(ValidationAccuracy.split(',')[0])))
print ("Testing Dataset Accuracy    = " + str(float(TestAccuracy.split(',')[0])))
print()
W_Now        = np.dot(220, W)
La           = 2 # Regularization Term: added to prevent over-fitting
learningRate = 0.02


for i in range(0,1266):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i])
    # ∇ ED = −(tn −w(τ)'φ(xn))φ(xn)
    La_Delta_E_W  = np.dot(La,W_Now)
    Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    # ∇ E = ∇ ED + λ*∇ EW
    # Stochastic Gradient Descent Solution for w
    
    #-----------------TrainingData Accuracy---------------------#
TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
    
    #-----------------ValidationData Accuracy---------------------#
VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
    
    #-----------------TestingData Accuracy---------------------#
TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
Erms_Test = GetErms(TEST_OUT,TestDataAct)

print ('----------Gradient Descent Solution-----------------')
print ('----------------Feature Subtraction---------------')
print ("E_rms Training   = " + str(Erms_TR.split(',')[1]))
print ("E_rms Validation = " + str(Erms_Val.split(',')[1]))
print ("E_rms Testing    = " + str(Erms_Test.split(',')[1]))
print ("Training Dataset Accuracy   = " + str(Erms_TR.split(',')[0]))
print ("Validation Dataset Accuracy = " + str(Erms_Val.split(',')[0]))
print ("Testing Dataset Accuracy    = " + str(Erms_Test.split(',')[0]))
print()

# In[25]:


W_Now        = np.zeros(472)
La           = 0.5# Regularization Term: added to prevent over-fitting
learningRate = 0.001

#for step in range(0,1266):
#   scores = np.dot(W_Now,(TrainingS.T[i]))
#   predictions = sigmoid(scores)

    # Update weights with gradient
#    output_error_signal = TrainingTarget - predictions
#    gradient = np.dot((TrainingS), output_error_signal)
#    W_Now += La * gradient
        
for i in range(0,3359):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    scores = np.dot(W_Now,TrainingS)
    predictions = sigmoid(scores)
    # Update weights with gradient
    output_error_signal = -(TrainingTarget - predictions)
    Delta_E_D     = np.dot((TrainingS), output_error_signal)/ TrainingTarget.size
    #La_Delta_E_W  = np.dot(La,W_Now)
    #Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E_D)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    # ∇ E = ∇ ED + λ*∇ EW
    # Stochastic Gradient Descent Solution for w
    
print("Logistic Regression")
print ('Feature Subtraction')
    
    #-----------------TrainingData Accuracy---------------------#
right=0
wrong=0
TR_TEST=[]
TR_TEST_OUT   = sigmoid(np.dot(W_Now,TrainingS))
for data in TR_TEST_OUT:
    if (data>0.5):
        TR_TEST.append(1)
    else:
        TR_TEST.append(0)  
for i,j in zip(TR_TEST,TrainingTarget):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("TrainingData Accuracy: " + str(right/(right+wrong)*100))
   
    #-----------------ValidationData Accuracy---------------------#
right=0
wrong=0
VAL_TEST=[]
VAL_TEST_OUT  = sigmoid(np.dot(W_Now,(ValDataS)))
for data in VAL_TEST_OUT:
    if (data>0.5):
        VAL_TEST.append(1)
    else:
        VAL_TEST.append(0) 
for i,j in zip(VAL_TEST,ValDataAct):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("ValidationData Accuracy: " + str(right/(right+wrong)*100))  
    
    #-----------------TestingData Accuracy---------------------#
right=0
wrong=0
TEST=[]
TEST_OUT      = sigmoid(np.dot(W_Now,(TestDataS)))
for data in TEST_OUT:
    if (data>0.5):
        TEST.append(1)
    else:
        TEST.append(0) 
for i,j in zip(TEST,TestDataAct):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("TestingData Accuracy: " + str(right/(right+wrong)*100))  
print()

# In[26]:


W_Now        = np.zeros(935)
La           = 0.5# Regularization Term: added to prevent over-fitting
learningRate = 0.001

#for step in range(0,1266):
#   scores = np.dot(W_Now,(TrainingS.T[i]))
#   predictions = sigmoid(scores)

    # Update weights with gradient
#    output_error_signal = TrainingTarget - predictions
#    gradient = np.dot((TrainingS), output_error_signal)
#    W_Now += La * gradient
        
for i in range(0,3359):
    
    #print ('---------Iteration: ' + str(i) + '--------------')
    scores = np.dot(W_Now,TrainingC)
    predictions = sigmoid(scores)
    # Update weights with gradient
    output_error_signal = -(TrainingTarget - predictions)
    Delta_E_D     = np.dot((TrainingC), output_error_signal)/ TrainingTarget.size
    #La_Delta_E_W  = np.dot(La,W_Now)
    #Delta_E       = np.add(Delta_E_D,La_Delta_E_W)    
    Delta_W       = -np.dot(learningRate,Delta_E_D)
    W_T_Next      = W_Now + Delta_W
    W_Now         = W_T_Next
    # ∇ E = ∇ ED + λ*∇ EW
    # Stochastic Gradient Descent Solution for w
    
print("Logistic Regression")
print ('Feature Concatenation') 
print()      
    #-----------------TrainingData Accuracy---------------------#
right=0
wrong=0
TR_TEST=[]
TR_TEST_OUT   = sigmoid(np.dot(W_Now,TrainingC))
for data in TR_TEST_OUT:
    if (data>0.5):
        TR_TEST.append(1)
    else:
        TR_TEST.append(0)  
for i,j in zip(TR_TEST,TrainingTarget):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("TrainingData Accuracy: " + str(right/(right+wrong)*100))
   
    #-----------------ValidationData Accuracy---------------------#
right=0
wrong=0
VAL_TEST=[]
VAL_TEST_OUT  = sigmoid(np.dot(W_Now,(ValDataC)))
for data in VAL_TEST_OUT:
    if (data>0.5):
        VAL_TEST.append(1)
    else:
        VAL_TEST.append(0) 
for i,j in zip(VAL_TEST,ValDataAct):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("ValidationData Accuracy: " + str(right/(right+wrong)*100))  
    
    #-----------------TestingData Accuracy---------------------#
right=0
wrong=0
TEST=[]
TEST_OUT      = sigmoid(np.dot(W_Now,(TestDataC)))
for data in TEST_OUT:
    if (data>0.5):
        TEST.append(1)
    else:
        TEST.append(0) 
for i,j in zip(TEST,TestDataAct):
    if i==j:
        right = right + 1
    else:
        wrong = wrong + 1
print("TestingData Accuracy: " + str(right/(right+wrong)*100))  


# In[29]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras.utils import np_utils
import numpy as np

TrainingTarget1= np_utils.to_categorical(np.array(TrainingTarget),2)


# In[34]:


input_size = 472
drop_out = 0.1
first_dense_layer_nodes  = 944
second_dense_layer_nodes = 2


def get_model():

    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))
    
    model.add(Dense(first_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
   
    model.summary()
    
    # Why use categorical_crossentropy?
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[35]:


model = get_model()


# In[36]:


validation_data_split = 0.0
num_epochs = 100
model_batch_size = 64
tb_batch_size = 32
early_patience = 100

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=0, patience=early_patience, mode='min')
history = model.fit(np.transpose(TrainingS)
                    , TrainingTarget1
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# In[45]:

print ()
print ("Neural Network")
print ("Feature Subtraction")
print()
right=0
wrong=0
predictedTestLabel=[]
for i,j in zip(np.transpose(TestDataS),TestDataAct):
    y = model.predict(np.array(i).reshape(-1,472))
    
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Accuracy: " + str(right/(right+wrong)*100))


# In[38]:


input_size = 935
drop_out = 0.2
first_dense_layer_nodes  = 1870
second_dense_layer_nodes = 2


def get_model_again():

    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))
    
    model.add(Dense(first_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
   
    model.summary()
    
    # Why use categorical_crossentropy?
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# In[39]:


model_again = get_model_again()


# In[40]:


validation_data_split = 0.0
num_epochs = 100
model_batch_size = 128
tb_batch_size = 32
early_patience = 10

tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=2, patience=early_patience, mode='min')
history = model_again.fit(np.transpose(TrainingC)
                    , TrainingTarget1
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# In[43]:

print ()
print ("Neural Network")
print ("Feature Concatenation")
print()
right=0
wrong=0
predictedTestLabel=[]
for i,j in zip(np.transpose(TestDataC),TestDataAct):
    y = model_again.predict(np.array(i).reshape(-1,935))
    predictedTestLabel.append(y.argmax())
    
    if j == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Accuracy: " + str(right/(right+wrong)*100))
print ()

