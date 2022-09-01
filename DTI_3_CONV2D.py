# -*- coding: utf-8 -*-
"""
@author: betsa
"""
#!pip install keras_tuner

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import csv
import pdb
import keras_tuner as kt
from keras_tuner import HyperParameter, HyperParameters
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from keras_tuner import Objective
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
#-------------------------------------------
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
def create_Interaction_mat(Win_s,ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_matrix,Y_Train,Y_Test,Drug_S,Prot_S):
    N_Tr=len(ind_d_Tr)
    N_Te=len(ind_d_Te)
    Temp_Y_matrix=np.zeros(np.shape(Y_matrix))
    #Remove Test values from Y
    for i in range(0,N_Tr):
        Temp_Y_matrix[ind_d_Tr[i]][ind_p_Tr[i]]=Y_matrix[ind_d_Tr[i]][ind_p_Tr[i]]

    # sort drug and protein based on similarity
    S1=np.array(Drug_S)
    S2=np.array(Prot_S)
    Sorted_drug_ind=np.argsort(S1)
    Sorted_prot_ind=np.argsort(S2)
    Win_sd=Win_s[0]
    Win_sp=Win_s[1]
    X_Train=[]
    X_Test=[]
    #fig=plt.pyplot.figure(figsize=(12,4))
    for i in range(0,N_Tr):
        ind1=ind_d_Tr[i]
        ind1=Sorted_drug_ind[ind1][-Win_sd:]
        ind2=ind_p_Tr[i]
        ind2=Sorted_prot_ind[ind2][-Win_sp:]
        t1=Temp_Y_matrix[ind1][:]
        temp_Y=t1[:,ind2]
        temp_Y=np.array(temp_Y)
        temp_Y1=temp_Y
        temp_Y1[-1][-1]=0
        avg=np.mean(temp_Y[np.nonzero(temp_Y)])
        index=np.argwhere(temp_Y==0)
        for item in index:
            temp_Y1[item[0]][item[1]]=avg

        X_Train.append(temp_Y1)

        #---------
    for i in range(0,N_Te):
        ind1=ind_d_Te[i]
        ind1=Sorted_drug_ind[ind1][-Win_sd:]
        ind2=ind_p_Te[i]
        ind2=Sorted_prot_ind[ind2][-Win_sp:]
        t1=Temp_Y_matrix[ind1][:]
        temp_Y=t1[:,ind2]
        temp_Y=np.array(temp_Y)
        avg=np.mean(temp_Y[np.nonzero(temp_Y)])
        index=np.argwhere(temp_Y==0)
        for item in index:
            temp_Y[item[0]][item[1]]=avg
        X_Test.append(temp_Y)
        '''if (i<3):
           ax=fig.add_subplot(2,3,i+1)
           ax.imshow(temp_Y)'''




    #plt.pyplot.show()
    #pdb.set_trace()

    X_Train=tf.convert_to_tensor(X_Train)
    X_Test=tf.convert_to_tensor(X_Test)


    return X_Train, X_Test
#------------------------------------------------------------
def create_sim_mat(Win_s,ind_Tr,ind_Te,Similarity):
    N_Tr=len(ind_Tr)
    N_Te=len(ind_Te)

    # sort  similarity
    S1=np.array(Similarity)
    Sorted_ind=np.argsort(S1)
    Win_1=Win_s[0]
    Win_2=Win_s[1]
    X_Train=[]
    X_Test=[]
    for i in range(0,N_Tr):
        ind1=ind_Tr[i]
        ind1=Sorted_ind[ind1][-Win_1:]
        t1=S1[ind1][:]
        temp_Y=t1[:,ind1]
        temp_Y=np.array(temp_Y)
        X_Train.append(temp_Y)
        #---------
    for i in range(0,N_Te):
        ind1=ind_Te[i]
        ind1=Sorted_ind[ind1][-Win_1:]
        t1=S1[ind1][:]
        temp_Y=t1[:,ind1]
        temp_Y=np.array(temp_Y)
        X_Test.append(temp_Y)




    #plt.pyplot.show()
    #pdb.set_trace()

    X_Train=tf.convert_to_tensor(X_Train)
    X_Test=tf.convert_to_tensor(X_Test)


    return X_Train, X_Test
#----------------------------------------------------
def cross_validation_Train_Test1(index_d,index_p,Y,num_fold,curr_fold,mode):
    N=len(index_d)
    fold_size=np.floor(N/num_fold)
    #print("fold_size=",fold_size)
    np.random.seed(8453)
    index=np.random.permutation(N)
    index=np.array(index)
    if mode=='Warm':
        low=np.int64((curr_fold-1)*fold_size)
        high=np.int64(curr_fold*fold_size)
        if curr_fold==num_fold:
          high=N
        Test_ind=index[low:high]
        Train_ind=np.setdiff1d(range(0,N) ,Test_ind)
        Test_ind=np.array(Test_ind)
        Train_ind=np.array(Train_ind)
        print('Train==',len(Train_ind))
        print('Test==',len(Test_ind))
        index_d=np.array(index_d)
        index_p=np.array(index_p)
        ind_d_Tr=index_d[Train_ind]
        ind_p_Tr=index_p[Train_ind]
        #print("len train drug",len(ind_d_Tr))
        ind_d_Te=index_d[Test_ind]
        ind_p_Te=index_p[Test_ind]
        #print("len test drug",len(ind_d_Te))
        Y=np.array(Y)
        Y_Train=Y[Train_ind]
        Y_Test=Y[Test_ind]
        #print('len Y_test',len(Y_Test))
        #print('len Y_train',len(Y_Train))
        pdb.set_trace()





    return ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test
#-------------------------------------------
class con_layer:
    def __init__(self,num_con,num_filter,kernel_size,win_size):
        self.num_con=num_con
        self.num_filter=num_filter
        self.win_size=win_size
        self.kernel_size=np.zeros(shape=(num_con,2),dtype=np.int32)
        for i in range(num_con):
           self.kernel_size[i][0]=kernel_size[i][0]
           self.kernel_size[i][1]=kernel_size[i][1]
#------------------------------------------
def NN_model1(hp):
  # Inputs
    Input_D=tf.keras.layers.Input(shape=(15,15,1))
    XD=Input_D
    
    #
    if option=='Interaction':
      Max_kernel=np.array([9,7,3])
      filter=hp.Choice('num_filter',values=[4,8,16,32])
      num_con=hp.Choice('num_Conv_layer',values=[1,2,3])
      for i in range(num_con):
           kernel_size=(hp.Int('kenerl'+str(i)+'_0',min_value=1, max_value=Max_kernel[i],step=2),
                   hp.Int('kenerl'+str(i)+'_1',min_value=1, max_value=Max_kernel[i],step=2))
           XD=tf.keras.layers.Conv2D(filters=filter*(2**i),
                                  kernel_size=kernel_size,
                                  activation='linear',padding='same',
                                  name= 'Drug_layer'+str(i), strides=(1,1))(XD)
           XD=tf.keras.layers.Dropout(0.2)(XD)
           XD=tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2))(XD)
    else:
      Max_kernel=np.array([9,7,3])
      filter=hp.Choice('num_filter',values=[4,8,16,32])
      num_con=hp.Choice('num_Conv_layer',values=[1,2,3])
      for i in range(num_con):
           hp.kernel_size=hp.Int('kenel_size'+str(i),min_value=1, max_value=Max_kernel[i],step=2)
           kernel_size=(hp.kernel_size,hp.kernel_size)
           XD=tf.keras.layers.Conv2D(filters=filter*(2**i),
                                  kernel_size=kernel_size,
                                  activation='linear',padding='same',
                                  name= 'Drug_layer'+str(i), strides=(1,1))(XD)
           XD=tf.keras.layers.Dropout(0.2)(XD)
           XD=tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2))(XD)

        
    XD=tf.keras.layers.Flatten()(XD)

    #--------------------------------------------------'''
    #concatenate
    X=XD
    num_hidden=hp.Choice('num_hidden',values=[1,2,3])
    num_units=hp.Choice('num_units',values=[64,128,256])

    for i in range(0,num_hidden):
        X=tf.keras.layers.Dense(units=num_units,activation='linear',
                                name='Dense_layer'+str(i))(X)
        X=tf.keras.layers.Dropout(rate=0.7)(X)
        num_units=np.floor(num_units/2)

    #-------------------------------------------------
    #output

    prediction=tf.keras.layers.Dense(units=1,activation=None)(X)
    #-------------------------------------------------
    #Model
    mdl=tf.keras.Model(inputs=Input_D,outputs=[prediction])
    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=['mse',tf.keras.metrics.RootMeanSquaredError()])
    return mdl
#-------------------------------------------
def set_parameters(index_d,index_p,Y,Y_matrix):
  # parameters
    global option
    fold=1 #second fold
    epochs=2
    ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                            index_p,Y,5,fold,'Warm')
    win_size=np.array([15,15]) 
    print('Drug Data preparation')
    Input_D_Tr,Input_D_Te=create_sim_mat(win_size,ind_d_Tr,ind_d_Te,Drug_S)
    print('portein Data preparation')
    Input_P_Tr,Input_P_Te=create_sim_mat(win_size,ind_p_Tr,ind_p_Te,Prot_S)
    print('Interaction Data preparation')
    Input_I_Tr,Input_I_Te=create_Interaction_mat(win_size,ind_d_Tr,
                                                     ind_p_Tr,ind_d_Te,ind_p_Te,
                                                     Y_matrix,Y_Train,Y_Test,
                                                     Drug_S,Prot_S)
  
    #--------------------------------------------
    #--------------------------------------------
    # find best Prot parameters
    print('===============Run Prot Hyperparam==============')
    X_Train=Input_P_Tr
    X_Test=Input_P_Te
    option='Prot'
    tuner2=RandomSearch(NN_model1,objective="val_mse",max_trials=40,overwrite=True)
    tuner2.search(X_Train,Y_Train,epochs=6,validation_data=(X_Test,Y_Test))
    #pdb.set_trace()
    #--------------------------------------------
    # find best drug parameters
    print('===============Run Drug Hyperparam==============')
    X_Train=Input_D_Tr
    X_Test=Input_D_Te
    option='Drug'
    tuner1=RandomSearch(NN_model1,objective="val_mse",max_trials=40,overwrite=True)
    tuner1.search(X_Train,Y_Train,epochs=6,validation_data=(X_Test,Y_Test))
    #pdb.set_trace()
    #--------------------------------------------
    # find best Interaction parameters
    print('===============Run Interaction Hyperparam==============')
    X_Train=Input_I_Tr
    X_Test=Input_I_Te
    option='Interaction'
    tuner3=RandomSearch(NN_model1,objective="val_mse",max_trials=60,overwrite=True)
    tuner3.search(X_Train,Y_Train,epochs=6,validation_data=(X_Test,Y_Test))
    #--------------------------------------------
    #--------------------------------------------
    print("***********************Best HyperParameters*********************************")
    print('--------Drug network properties:')
    best_hps_drug=tuner1.get_best_hyperparameters()[0]
    print(best_hps_drug.values)

    print('-------Interaction network properties:')
    best_hps_inter=tuner3.get_best_hyperparameters()[0]
    print(best_hps_inter.values)

    print('-------Prots network properties:')
    best_hps_prot=tuner2.get_best_hyperparameters()[0]
    print(best_hps_prot.values)
  
    temp1=np.array([best_hps_drug['num_hidden'],best_hps_prot['num_hidden'],
                   best_hps_inter['num_hidden']])
    #print(temp1)
    temp2=np.array([best_hps_drug['num_units'],best_hps_prot['num_units'],
                   best_hps_inter['num_units']])
    #print(temp2)
    num_hidden=np.max(temp1)
    num_units=np.max(temp2)
    print('--------Denese layer properties:')
    print('num hidden layers=',num_hidden)
    print('num units=',num_units)
    #----------------------------------------------------
    num_conn_d=best_hps_drug['num_Conv_layer']
    print(type(num_conn_d))
    K_d=np.zeros(shape=(num_conn_d,2),dtype=np.int32)
    for i in range(num_conn_d):
      name='kenel_size'+str(i)
      K_d[i][0]=best_hps_drug[name]
      K_d[i][1]=best_hps_drug[name]
  
    drug_info=con_layer(num_conn_d,best_hps_drug['num_filter'],
                        K_d,(15,15))
    
    num_conn_p=best_hps_prot['num_Conv_layer']
    K_p=np.zeros(shape=(num_conn_p,2),dtype=np.int32)
    for i in range(num_conn_p):
      name='kenel_size'+str(i)
      K_p[i][0]=best_hps_prot[name]
      K_p[i][1]=best_hps_prot[name]
    prot_info=con_layer(num_conn_p,best_hps_prot['num_filter'],K_p,(15,15))

    num_conn_i=best_hps_inter['num_Conv_layer']
    K_i=np.zeros(shape=(num_conn_p,2),dtype=np.int32)
    for i in range(num_conn_i):
      name='kenerl'+str(i)+'_0'
      K_i[i][0]=best_hps_inter[name]
      name='kenerl'+str(i)+'_1'
      K_i[i][1]=best_hps_inter[name]

    interaction_info=con_layer(num_conn_i,best_hps_inter['num_filter'],K_i,(15,15))
    pdb.set_trace()
    return drug_info,prot_info,interaction_info,num_hidden,num_units
    
#-------------------------------------------
def NN_model(drug_info,prot_info,interaction_info,num_hidden,num_units):
    # Inputs
    Input_D=tf.keras.layers.Input(shape=(drug_info.win_size[0],drug_info.win_size[1],1))
    Input_P=tf.keras.layers.Input(shape=(prot_info.win_size[0],prot_info.win_size[1],1))
    Input_I=tf.keras.layers.Input(shape=(interaction_info.win_size[0],interaction_info.win_size[1],1))

    XD=Input_D
    XP=Input_P
    XI=Input_I
    # Drug
    for i in range(drug_info.num_con):
        XD=tf.keras.layers.Conv2D(filters=drug_info.num_filter*(2**i),
                                  kernel_size=(drug_info.kernel_size[i][0],drug_info.kernel_size[i][1]),
                                  activation='linear',padding='same',
                                  name= 'Drug_layer'+str(i), strides=(1,1))(XD)
        XD=tf.keras.layers.Dropout(0.2)(XD)
        XD=tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2,2))(XD)
        

    #XD=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XD)
    XD=tf.keras.layers.Flatten()(XD)
    # Prot
    for i in range(prot_info.num_con):
        XP=tf.keras.layers.Conv2D(filters=prot_info.num_filter*(2**i),
                                  kernel_size=(prot_info.kernel_size[i][0],prot_info.kernel_size[i][1]),
                                  activation='linear',padding='same',
                                  name='Prot_layer'+str(i), strides=(1,1))(XP)
        XP = tf.keras.layers.Dropout(0.2)(XP)
        XP = tf.keras.layers.AveragePooling2D(strides=(2, 2), pool_size=(2, 2))(XP)

    #XP=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XP)
    XP=tf.keras.layers.Flatten()(XP)
    # Interaction
    for i in range(interaction_info.num_con):
        XI=tf.keras.layers.Conv2D(filters=interaction_info.num_filter*(2**i),
                                  kernel_size=(interaction_info.kernel_size[i][0],interaction_info.kernel_size[i][1]),
                                  activation='linear',padding='same',
                                  strides=(1,1))(XI)
        XI=tf.keras.layers.Dropout(0.2) (XI)
        XI=tf.keras.layers.AveragePooling2D(strides=(2,2),pool_size=(2, 2))(XI)

    #XI=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XI)
    XI=tf.keras.layers.Flatten()(XI)

    #--------------------------------------------------'''
    #concatenate
    X=tf.keras.layers.concatenate([XD,XP,XI])
  

    for i in range(0,num_hidden):
        X=tf.keras.layers.Dense(units=num_units,activation='linear',name='Dense_layer'+str(i))(X)
        X=tf.keras.layers.Dropout(rate=0.7)(X)
        num_units=np.floor(num_units/2)

    #-------------------------------------------------
    #output

    prediction=tf.keras.layers.Dense(units=1,activation=None)(X)
    #-------------------------------------------------
    #Model
    mdl=tf.keras.Model(inputs=[Input_D,Input_P,Input_I],outputs=[prediction])
    mdl.summary()
    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return mdl
# -------------------------------------------
def Model_Training (mdl,X_Train,Y_Train,X_Test,Y_Test):
  epochs=100
  print('=================== Fit the model ======================')
  pdb.set_trace()
  Train_result=mdl.fit(X_Train,Y_Train,epochs=epochs,verbose=2)
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
  #print('-------------Test Result--------------------')
  #Y_Test=np.array(Y_Test)
  #Test_result=mdl.evaluate(x=([Input_D_Te,Input_P_Te,Input_I_Te]),y=Y_Test,verbose=2)
  #print('-----------------------------------')
  #print('---------------Result--------------')
  #print(Test_result)
  # Test
  Y_hat1=mdl.predict(X_Test)
  #print(np.shape(Y_hat1))
  Y_Test=np.reshape(Y_Test,np.shape(Y_hat1))
  MSE_Test=np.mean((Y_Test-Y_hat1)**2)
  RMSE_Test=np.sqrt(MSE_Test)
  print('---------------MY Test Errors------------------')
  print('MSE=',MSE_Test,'RMSE=',RMSE_Test)
  print('------------individual Test error first 10 data-------------')
  t=0
  for i in range(len(Y_Test)):
    er=(Y_Test[i]-Y_hat1[i])**2
    if i<10:
        print("\nTure=",np.round(Y_Test[i],3))
        print("\nPredict=",np.round(Y_hat1[i],3))
        print("\nerror=",np.round(er,3))
        print('\n******************************************')
    t=er+t
  print('========================================')  
  print("manual RMSE Test=", np.sqrt(t/len(Y_Test)))
  #Train
  Y_hat2=mdl.predict(X_Train)
  Y_Train=np.array(Y_Train)
  Y_Train=np.reshape(Y_Train,np.shape(Y_hat2))
  MSE_Train=np.mean((Y_Train-Y_hat2)**2)
  RMSE_Train=np.sqrt(MSE_Train)
  print('----------------- MY Train Errors------------------')
  print('MSE=',MSE_Train,'RMSE=',RMSE_Train)

  ax=fig.add_subplot(1,3,3)
  ax.plot(Y_hat1,Y_Test,'o')
  ax.plot(range(4,11),range(4,11),'r:o')
  ax.set_xlabel('Predicted label',size=10)
  ax.set_ylabel('True label',size=10)
  plt.pyplot.show()
  return MSE_Train, RMSE_Train, MSE_Test,RMSE_Test
#----------------------------------------------
def CV_evaluation(drug_info,prot_info,interaction_info,num_hidden,num_units,index_d,index_p,Y,Y_matrix ,num_fold,Drug_S,Prot_S):
    # parameters
    folds=range(1,num_fold+1)
    print(folds)
    All_MSE_Test=[]
    All_RMSE_Test=[]
    for i in folds:
        print("===============================fold ",i,"=================================")
        ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                                                                                        index_p,Y,num_fold,i,'Warm')


        print('Drug Data preparation')
        Input_D_Tr,Input_D_Te=create_sim_mat(drug_info.win_size,ind_d_Tr,ind_d_Te,Drug_S)
        #print('drug train',len(Input_D_Tr))
        #print('drug test',len(Input_D_Te))
        print('portein Data preparation')
        Input_P_Tr,Input_P_Te=create_sim_mat(prot_info.win_size,ind_p_Tr,ind_p_Te,Prot_S)
        #print('prot train',len(Input_P_Tr))
        #print('prot test',len(Input_P_Te))
        print('Interaction Data preparation')
        Input_I_Tr,Input_I_Te=create_Interaction_mat(interaction_info.win_size,ind_d_Tr,
                                                     ind_p_Tr,ind_d_Te,ind_p_Te,
                                                     Y_matrix,Y_Train,Y_Test,
                                                     Drug_S,Prot_S)
        #print('Inter train',len(Input_I_Tr))
        #print('Inter test',len(Input_I_Te))
        
        
        X_Train=([Input_D_Tr,Input_P_Tr,Input_I_Tr])
        X_Test=([Input_D_Te,Input_P_Te,Input_I_Te])
        print(len(X_Train))
        mdl=NN_model(drug_info,prot_info,interaction_info,num_hidden,num_units)
        MSE_Train, RMSE_Train, MSE_Test,RMSE_Test=Model_Training (mdl,X_Train,Y_Train,X_Test,Y_Test)
        All_MSE_Test.append(MSE_Test)
        All_RMSE_Test.append(RMSE_Test)
        pdb.set_trace()
    #--------------------------------------------------
    All_MSE_Test=np.array(All_MSE_Test)
    AVG_5folf_MSE=np.mean(All_MSE_Test)
    Std_5folf_MSE=np.std(All_MSE_Test)
    All_RMSE_Test=np.array(All_RMSE_Test)
    AVG_5folf_RMSE=np.mean(All_RMSE_Test)
    Std_5folf_RMSE=np.std(All_RMSE_Test)
    print('========================5 fold results===========================')
    print('MSE :Avg+std=',AVG_5folf_MSE,'+',Std_5folf_MSE)
    print('RMSE: Avg+std=',AVG_5folf_RMSE,'+',Std_5folf_RMSE)




#-------------------------------------------
#-------------------------------------------
#-------------------------------------------
# 1-read dataset
#path='E:\\PhD_thesis\\research\\semisuppervised\\drug_target\\thesis_code\\negative_examples\\checked_codes\\upload\\Code\\paper_2_code\\upload\\'
f_name='Davis.csv'
Prot_name,Drug_id,KD_val=read_data(f_name)
# 2-read similarities
f_name='Davis_Drug_Fingerprint_sim.csv'
All_D_id,Drug_S=read_simarity(f_name)
f_name='Davis_Prot_SW_sim.csv'
All_P_id,Prot_S=read_simarity(f_name)
# 3-create dataset
index_d,index_p,Y,Y_matrix =create_dataset(Prot_name, Drug_id, KD_val,All_D_id,All_P_id)

# 4- create model
drug_info=con_layer(2,16,[(3,3)],(15,15))
prot_info=con_layer(1, 32, [(3,3)],(15,15))
interaction_info=con_layer(1, 64, [(9,7)],(15,15))
num_hidden=2
num_units=256
#drug_info,prot_info,interaction_info,num_hidden,num_units=set_parameters(index_d,index_p,Y,Y_matrix)
# 5- 5-CV evaluation
CV_evaluation(drug_info,prot_info,interaction_info,num_hidden,num_units,index_d,index_p,Y,Y_matrix ,5,Drug_S,Prot_S)
