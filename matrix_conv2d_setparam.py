# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:53:43 2022

@author: betsa
"""


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import csv
import pdb
from kerastuner import HyperParameters, HyperParameter
from kerastuner.tuners import RandomSearch

#----------------------------------------
def read_data(file_name):
    f=open(file_name)
    data=csv.reader(f)
    Drug_id=next(data)
    Drug_id=Drug_id[4:]
    next(data)
    next(data)
 #   print(Drug_id)
    Prot_name=[]
    KD_val=[]
    for r in data:
        Prot_name.append(r[2])
        temp=r[4:]
        temp1=[]
        for t in temp:
            if t=='':
               temp1.append(10000) 
            else:
               temp1.append(float(t))
               
        KD_val.append(temp1)
       
        
    return Prot_name, Drug_id, KD_val
#--------------------------------------------
class interaction:
    def __init__(self,d_id,p_id,feature):
        self.d_id=d_id
        self.p_id=p_id
        self.feature=feature
        
#-------------------------------------------
def create_dataset(Prot_name, Drug_id, KD_val,All_D_id,All_P_id):
    num_drug=len(All_D_id)
    num_prot=len(All_P_id)
    index_d=[]
    index_p=[]
    Y_matrix=np.zeros(shape=(num_drug,num_prot))
    Y=[]
    KD_val=np.array(KD_val)
    t=KD_val/(10**9)
    KD_val=-1*np.log10(t)
    for i in range(num_drug):
        temp=[]
        #s1=np.array(Drug_S[index_d])
        for j in range(num_prot):
            ind1=All_D_id.index(Drug_id[i])
            index_d.append(ind1)
            ind2=All_P_id.index(Prot_name[j])
            index_p.append(ind2)
            #s2=np.array(Prot_S[index_p])
            #feature=np.concatenate((s1,s2))
            # a=interaction(index_d, index_p,feature)
            #X.append(feature)
            temp.append(KD_val[j][i])
            Y.append(KD_val[j][i])
            Y_matrix[ind1][ind2]=KD_val[j][i]
        '''if (len(Y)>5000):
                break'''

   
    return index_d,index_p,Y,Y_matrix   
#-------------------------------------------
def read_simarity(file_name):
   f=open(file_name)
   Data=csv.reader(f)
   ID=next(Data)
   N=len(ID)
   i=0
   S=tf.zeros(shape=(N,N),dtype=tf.float64)
   for temp in Data:
       # j=0
        #for t2 in temp:
       v=np.float64(temp)
       S=tf.tensor_scatter_nd_update(S, [[i,]], [v])
          #  j=j+1
       i=i+1    
            
            
   return ID, S
#--------------------------------------------
def create_Features(Win_s,ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_matrix,Y_Train,Y_Test,Drug_S,Prot_S):
 N_Tr=len(ind_d_Tr)
 N_Te=len(ind_d_Te)
 #Remove Test values from Y
 for i in range(0,N_Te):
     Y_matrix[ind_d_Te[i]][ind_p_Te[i]]=0
     
 # sort drug and protein based on similarity
 S1=np.array(Drug_S)
 S2=np.array(Prot_S)
 Sorted_drug_ind=np.argsort(S1)
 Sorted_prot_ind=np.argsort(S2)
 Win_sd=Win_s[0]
 Win_sp=Win_s[1]
 X_Train=[]
 X_Test=[]
 for i in range(0,N_Tr):
     ind1=ind_d_Tr[i]
     ind1=Sorted_drug_ind[ind1][-Win_sd:]
     ind2=ind_p_Tr[i]
     ind2=Sorted_prot_ind[ind2][-Win_sp:]
     t1=Y_matrix[ind1][:]
     temp_Y=t1[:,ind2]
     temp_Y=np.array(temp_Y)
     temp_Y[-1][-1]=0
     avg=np.mean(temp_Y[np.nonzero(temp_Y)])
     index=np.argwhere(temp_Y==0)
     for item in index:
         temp_Y[item[0]][item[1]]=avg

     X_Train.append(temp_Y)  
     
     #---------   
 for i in range(0,N_Te):
    ind1=ind_d_Te[i]
    ind1=Sorted_drug_ind[ind1][-Win_sd:]
    ind2=ind_p_Te[i]
    ind2=Sorted_prot_ind[ind2][-Win_sp:]
    t1=Y_matrix[ind1][:]
    temp_Y=t1[:,ind2]
    temp_Y=np.array(temp_Y)
    avg=np.mean(temp_Y[np.nonzero(temp_Y)])
    index=np.argwhere(temp_Y==0)
    for item in index:
        temp_Y[item[0]][item[1]]=avg
    X_Test.append(temp_Y)
    
     
     
     
 
 X_Train=tf.convert_to_tensor(X_Train)
 dX=tf.data.Dataset.from_tensor_slices(X_Train)
 dY=tf.data.Dataset.from_tensor_slices(Y_Train)
 Data_Train=tf.data.Dataset.zip((dX,dY))

 X_Test=tf.convert_to_tensor(X_Test)
 dX=tf.data.Dataset.from_tensor_slices(X_Test)
 dY=tf.data.Dataset.from_tensor_slices(Y_Test)
 Data_Test=tf.data.Dataset.zip((dX,dY))    
       
 return X_Train,X_Test,Data_Train,Data_Test
#-------------------------------------------
def cross_validation_Train_Test1(index_d,index_p,Y,num_fold,curr_fold,mode):
    N=len(index_d)
    fold_size=np.floor(N/num_fold)
    np.random.seed(845)
    index=np.random.permutation(N)
    index=np.array(index)
    if mode=='Warm':
        low=np.int64((curr_fold-1)*fold_size)
        high=np.int64(curr_fold*fold_size)
        Test_ind=index[low:high]
        Train_ind=np.setdiff1d(range(0,N) ,Test_ind)
        Test_ind=np.array(Test_ind)
        Train_ind=np.array(Train_ind)
        index_d=np.array(index_d)
        index_p=np.array(index_p)
        ind_d_Tr=index_d[Train_ind]
        ind_p_Tr=index_p[Train_ind]
        ind_d_Te=index_d[Test_ind]
        ind_p_Te=index_p[Test_ind]
        Y=np.array(Y)
        Y_Train=Y[Train_ind]
        Y_Test=Y[Test_ind]
        
        
   
    
    
    return ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test  
#----------------------------------------------------------------
def set_hyparam(index_d,index_p,Y,Y_matrix,num_fold,Drug_S,Prot_S):
    #parameters
    epochs=10
    batch_size=100
    W_size=[11,15,20,25]
    K=[1,3,5,7,9]
    Param=[]
    error=[]
    iter=0
    #---------------------------
    ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                                                             index_p,Y,num_fold,1,'Warm')
    for w in W_size:
        win_size=np.array([w,w])
        Data_Train,Data_Test=create_Features(win_size,ind_d_Tr,ind_p_Tr,
                                             ind_d_Te,ind_p_Te,Y_matrix,
                                             Y_Train,Y_Test,Drug_S,Prot_S)
        Data_Train=tf.data.Dataset.batch(Data_Train, batch_size=batch_size)
        Data_Test=tf.data.Dataset.batch(Data_Test, batch_size=len(Data_Test))
        for K0 in K:
          for K1 in K:
            print("======================Iteration %d======================"%(iter))
            P=[w,K0,K1]
            print("Params=",P)
            kernel_size=(K0,K1)
            mdl=NN_model(64,kernel_size,1,1,64,win_size)
            Train_result=mdl.fit(Data_Train,epochs=epochs,verbose=1)
            Test_result=mdl.evaluate(Data_Test,verbose=1)
            error.append(Test_result)
            Param.append(P)
            print("**********************************************")
            iter=iter+1
    
    error=np.array(error)
    Param=np.array(Param) 
    pdb.set_trace() 
    best_ind=np.argmin(error[:,1])
    best_param=Param[best_ind]
    print('BEST PARAM:',best_param)
    print('best_ind',best_ind)
    kernel_size=(best_param[1],best_param[2])
    win_size=np.array([best_param[0],best_param[0]])
    mdl=NN_model(64,kernel_size,1,1,64,win_size)
    mdl.summary()
    pdb.set_trace()
    return mdl   

#-------------------------------------------
def NN_model(num_filt,kernel_size,num_con,num_hidden,num_units,win_size):
    mdl=tf.keras.Sequential()
    Dp1=0.2
        #initializer = tf.keras.initializers.Ones()
    K0=[7,3,3]
    K1=[7,5,1]
    #----------------------------------------
    # First layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt,
                                   kernel_size=(K0[0],K1[0]),
                                   activation='linear',padding='same',
                                   strides=(1,1)))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2)))
    # Second layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt*2,
                                   kernel_size=(K0[1],K1[1]),
                                   activation='elu',padding='same'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2)))
    # third layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt*4,
                                   kernel_size=(K0[2],K1[2]),
                                   activation='linear',padding='same'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.GlobalAveragePooling2D())  
    
    #mdl.add(tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2)))
    mdl.add(tf.keras.layers.Flatten())
    #mdl.add(tf.keras.layers.Conv2D(filters=num_filt,kernel_size=kernel_size,
                                  # activation='relu',padding='same',
                                  # strides=(1,1)))
    #mdl.add(tf.keras.layers.GlobalAveragePooling2D())
    #-------------------------------------------------
    Dp2=0.7
    # first Dense layer
    mdl.add(tf.keras.layers.Dense(units=num_units,activation='relu'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp2))
       # third Dense layer
    mdl.add(tf.keras.layers.Dense(units=np.int64(num_units/2),activation='linear'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp2))
        
    
    
    mdl.add(tf.keras.layers.Dense(units=1,activation=None))
                                  
    mdl.build(input_shape=(None,win_size[0],win_size[1],1))
    mdl.summary()
    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse',tf.keras.metrics.RootMeanSquaredError()])
    return mdl
#---------------------------------------------------------------------------
#----------------------------------------------------------------
def set_hyparam1(index_d,index_p,Y,Y_matrix,num_fold,Drug_S,Prot_S,win_size):
    #parameters
    epochs=10
    batch_size=100
    Param=[]
    error=[]
    iter=0
    #---------------------------
    ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                                                             index_p,Y,num_fold,1,'Warm')
    X_Train,X_Test,Data_Train,Data_Test=create_Features(win_size,ind_d_Tr,ind_p_Tr,
                                         ind_d_Te,ind_p_Te,Y_matrix,
                                         Y_Train,Y_Test,Drug_S,Prot_S)

    tuner=RandomSearch(build_NN_model,objective="val_mse",max_trials=70)
    tuner.search(X_Train,Y_Train,epochs=6,validation_data=(X_Test,Y_Test))
       
    pdb.set_trace() 
    print('=======================================================')
    best_param=tuner.get_best_hyperparameters()[0]
    print('BEST PARAM:',best_param.values)
    mdl=tuner.hypermodel.build(best_param)
    mdl.summary()
    pdb.set_trace()
    return mdl   

#---------------------------------------------------------------------------
def build_NN_model(hp):
    mdl=tf.keras.Sequential()
    num_con=3#hp.Choice('num_conv_layer',values=[1,2,3])
    Dp1=hp.Choice('Drop_out_conv',values=[0.2,0.5,0.7])
    num_filt=hp.Choice('num_filter_',values=[8,16,32,64])
    # First layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt,
                                   kernel_size=(hp.Choice('Kenel_0_layer_1',values=[1,3,5,7,9]),
                                                hp.Choice('Kenel_1_layer_1',values=[1,3,5,7,9])),
                                   activation='linear',padding='same',
                                   strides=(1,1)))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2)))
    # Second layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt*2,
                                   kernel_size=(hp.Choice('Kenel_0_layer_2',values=[1,3,5]),
                                                hp.Choice('Kenel_1_layer_2',values=[1,3,5])),
                                   activation='linear',padding='same'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2)))
    # third layer
    mdl.add(tf.keras.layers.Conv2D(filters=num_filt*4,
                                   kernel_size=(hp.Choice('Kenel_0_layer_3',values=[1,3]),
                                                hp.Choice('Kenel_1_layer_3',values=[1,3])),
                                   activation='linear',padding='same'))
    mdl.add(tf.keras.layers.Dropout(rate=Dp1))
    mdl.add(tf.keras.layers.GlobalAveragePooling2D())  
    
    #mdl.add(tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2,2)))
    mdl.add(tf.keras.layers.Flatten())
    num_hidden=hp.Choice('num_Dense_layer',values=[1,2,3,4])
    Dp2=hp.Choice('Drop_out_Dense',values=[0.2,0.5,0.7])
    num_units=hp.Choice('num_units_',values=[32,64,128,256])
    # first Dense layer
    for i in range (num_hidden):
      mdl.add(tf.keras.layers.Dense(units=num_units,
                                    activation=hp.Choice('activation_'+str(i),values=['linear','relu','elu'])))
      mdl.add(tf.keras.layers.Dropout(rate=Dp2))
      num_units=np.int64(num_units/2)
    # Second Dense layer
    #mdl.add(tf.keras.layers.Dense(units=np.int64(num_units/2),activation='linear'))
    #mdl.add(tf.keras.layers.Dropout(rate=Dp2))
        
    
    
    mdl.add(tf.keras.layers.Dense(units=1,activation=None))
                                  
    mdl.build(input_shape=(None,15,15,1))
    #mdl.summary()
    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse',tf.keras.metrics.RootMeanSquaredError()])
    return mdl    
#---------------------------------------------------------------------------
def CV_evaluation(index_d,index_p,Y,Y_matrix ,mdl,num_fold,Drug_S,Prot_S,win_size):
    # parameters
    epochs=250
    batch_size=100
    folds=range(1,num_fold+1)
    error=[]
    for i in folds:
        ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                                                             index_p,Y,num_fold,i,'Warm')
        X_Train,X_Test,Data_Train,Data_Test=create_Features(win_size,ind_d_Tr,ind_p_Tr,
                                             ind_d_Te,ind_p_Te,Y_matrix,
                                             Y_Train,Y_Test,Drug_S,Prot_S)

        #batch_size=np.floor(len(Data_Train)/5)
        Data_Train=tf.data.Dataset.batch(Data_Train, batch_size=batch_size)
       
        
        #print(mdl.get_weights())
        Train_result=mdl.fit(Data_Train,epochs=epochs,verbose=1)
        #print(mdl.get_weights())
        
        print('======================================================')
        hist=Train_result.history
        fig=plt.pyplot.figure(figsize=(12,6))
        ax=fig.add_subplot(1,3,1)
        ax.plot(hist['loss'])
        ax.set_title('Train loss')
        ax=fig.add_subplot(1,3,2)
        ax.plot(hist['root_mean_squared_error'])
        ax.set_title('RMSE Train')
       
        
        Data_Test=tf.data.Dataset.batch(Data_Test, batch_size=len(Data_Test))
        Test_result=mdl.evaluate(Data_Test,verbose=1)
        print('-------------Test Result--------------------')
        print(Test_result)
        # Test
        Y_hat1=mdl.predict(X_Test) 
        #print(np.shape(Y_hat1))
        #print(np.shape(Y_Test))
        Y_Test=np.array(Y_Test)
        Y_Test=np.reshape(Y_Test,np.shape(Y_hat1))
        error.append(np.sqrt(np.mean((Y_Test-Y_hat1)**2)))
        print('---------------MY Test RMSE Error------------------')
        print(error)
        #Train
        Y_hat2=mdl.predict(X_Train)
        Y_Train=np.array(Y_Train)
        Y_Train=np.reshape(Y_Train,np.shape(Y_hat2))
        error_train=(np.sqrt(np.mean((Y_Train-Y_hat2)**2)))
        print('-----------------MT Train RMSE Error------------------')   
        print(error_train)

        
        ax=fig.add_subplot(1,3,3)
        ax.plot(Y_hat1,Y_Test,'o')
        ax.plot(range(4,11),range(4,11),'r:o')
        ax.set_xlabel('Predicted label',size=10)
        ax.set_ylabel('True label',size=10)
        plt.pyplot.show()
        
        pdb.set_trace()
        
#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
# 1-read dataset
#path='E:\\PhD_thesis\\research\\semisuppervised\\drug_target\\thesis_code\\negative_examples\\checked_codes\\upload\\Code\\paper_2_code\\upload\\'
f_name='Davis.csv'
Prot_name,Drug_id,KD_val=read_data(f_name) 
# 2-read similarities
f_name='Davis_Drug_Fingerprint_sim.csv'#path+'Data\\Davis\\
All_D_id,Drug_S=read_simarity(f_name)
f_name='Davis_Prot_SW_sim.csv'#path+'Data\\Davis\\
All_P_id,Prot_S=read_simarity(f_name)
# 3-create dataset   
index_d,index_p,Y,Y_matrix =create_dataset(Prot_name, Drug_id, KD_val,All_D_id,All_P_id)
# set param
num_fold=5
win_size=np.array([15,15])
mdl=set_hyparam1(index_d,index_p,Y,Y_matrix,num_fold,Drug_S,Prot_S,win_size)
# 4- create model
'''num_filt=16
kernel_size=(7,7)
num_con=3
num_units=256
num_hidden=2
mdl=NN_model(num_filt,kernel_size,num_con,num_hidden,num_units,win_size)'''
# 5- 5-CV evaluation
CV_evaluation(index_d,index_p,Y ,Y_matrix,mdl,num_fold,Drug_S,Prot_S,win_size)
