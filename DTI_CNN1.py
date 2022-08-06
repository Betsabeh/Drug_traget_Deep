# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:02:59 2022

@author: betsa
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt
import csv
import pdb
#---------------------------------------
def read_data(file_name):
  f=open(file_name)
  data=f=csv.reader(f)
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
#---------------------------------------------------
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
#---------------------------------------------------
def create_dataset(Prot_name, Drug_id, KD_val,All_D_id,Drug_S,All_P_id,Prot_S):
    num_drug=len(Drug_id)
    num_prot=len(Prot_name)
    X=[]
    Y=[]
    for i in range(num_drug):
        index_d=All_D_id.index(Drug_id[i])
        s1=np.array(Drug_S[index_d])
        for j in range(num_prot):
            index_p=All_P_id.index(Prot_name[j])
            s2=np.array(Prot_S[index_p])
            feature=np.concatenate((s1,s2))
           # a=interaction(index_d, index_p,feature)
            X.append(feature)
            Y.append(KD_val[j][i])
        
    
    Y=-1*np.log10(Y)
    dX=tf.convert_to_tensor(X)
    dX=tf.data.Dataset.from_tensor_slices(dX)
    dY=tf.convert_to_tensor(Y)
    dY=tf.data.Dataset.from_tensor_slices(dY)
    Data_set=tf.data.Dataset.zip((dX,dY))         
    return Data_set,X,Y     
#---------------------------------------------------
def NN_model(num_f,num_Con_layer,num_fil,kernel_size,po_size,num_h_layer,num_h_unit):
  mdl=tf.keras.Sequential()
  for i in range(0,num_Con_layer):
    mdl.add(tf.keras.layers.Conv1D(filters=num_fil,
                                 kernel_size=kernel_size,strides=1,
                                 activation='relu',padding='same',
                                 data_format='channels_last'))
    #print(i,'th CNN layer ouptput shape:')
    #print(mdl.compute_output_shape(input_shape=(None,num_f,1)))
    mdl.add(tf.keras.layers.MaxPool1D(pool_size=po_size,strides=2))
    print(i,'th pooling layer output shape: ')
    print(mdl.compute_output_shape(input_shape=(None,num_f,1)))
    num_fil=num_fil/2

  mdl.add(tf.keras.layers.Flatten())
  #print('the Flatten layer output shape:')
  #print(mdl.compute_output_shape(input_shape=(None,num_f,1)))
  for i in range(0,num_h_layer):
    mdl.add(tf.keras.layers.Dense(units=num_h_unit,activation='relu',
                                  ))
    print(i,'th FC layer output shape:')
    print(mdl.compute_output_shape(input_shape=(None,num_f,1)))
    mdl.add(tf.keras.layers.Dropout(rate=0.5))
    num_h_unit=num_h_unit/2
  
  mdl.add(tf.keras.layers.Dense(units=1,activation='linear'))
  mdl.build(input_shape=(None,num_f,1))
  mdl.summary()
  mdl.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.RootMeanSquaredError()])
  return mdl
#---------------------------------------------------
def Create_Train_Test(Data_set,num_fold,curr_fold,mode):
    N=len(Data_set)
    fold_size=np.floor(N/num_fold)
    tf.random.set_seed(1)
    if mode=='Warm':
        low=np.int64((curr_fold-1)*fold_size)
        high=np.int64(curr_fold*fold_size)
        Test_ind=range(low,high)
        Train_ind=np.setdiff1d(range(0,N) ,Test_ind)
        X_Train=[]
        Y_Train=[]
        X_Test=[]
        Y_Test=[]
        i=0
        for item in Data_set:
            if i in Train_ind:
                X_Train.append(item[0].numpy())
                Y_Train.append(item[1].numpy())
            else:
                X_Test.append(item[0].numpy())
                Y_Test.append(item[1].numpy())
            i=i+1
            
    ''' X_Train=np.array(X_Train)
    Y_Train=np.array(Y_Train)
    X_Test=np.array(X_Test)
    Y_Test=np.array(Y_Test)'''
    dX=tf.convert_to_tensor(X_Train)
    dX=tf.data.Dataset.from_tensor_slices(dX)
    dY=tf.convert_to_tensor(Y_Train)
    dY=tf.data.Dataset.from_tensor_slices(dY)
    Data_Train=tf.data.Dataset.zip((dX,dY))    
    
    dX=tf.convert_to_tensor(X_Test)
    dX=tf.data.Dataset.from_tensor_slices(dX)
    dY=tf.convert_to_tensor(Y_Test)
    dY=tf.data.Dataset.from_tensor_slices(dY)
    Data_Test=tf.data.Dataset.zip((dX,dY))    
    
    
    return Data_Train,Data_Test#X_Train,Y_Train,X_Test,Y_Test  
#---------------------------------------------------
def CV_evaluation(Data_set,Y,mdl,num_fold,mode):
    # parameters
    epochs=1
    batch_size=64
    folds=range(1,num_fold+1)
    error=[]
    Data_set=Data_set.shuffle(len(Y))
    for i in folds:
        Data_Train, Data_Test=Create_Train_Test(Data_set,num_fold,i,'Warm')
        Data_Train=tf.data.Dataset.batch(Data_Train, batch_size=batch_size)
        bs = Data_Train._batch_size.numpy()
        print(bs)
        '''for j,(batch_x, batch_y) in enumerate(Data_Train):
            xprint(j, batch_x.shape, batch_y.numpy())'''
        Train_result=mdl.fit(Data_Train,epochs=epochs,batch_size=batch_size,
                             verbose=2)
  
        hist=Train_result.history
        # test
        Data_Test=tf.data.Dataset.batch(Data_Test,batch_size=len(Data_Test))
        bs = Data_Test._batch_size.numpy()
        print(bs)
        '''for j,(batch_x, batch_y) in enumerate(Data_Test):
            print(j, batch_x.shape, batch_y.numpy())'''
        Test_result=mdl.evaluate(Data_Test,verbose=2)
        print('-----------------------------------')
        print('---------------Result--------------')
        print(Test_result)
        # Test
        Y_hat1=mdl.predict(Data_Test)
        Y_Test=[]
        for item in Data_Test:
            Y_Test.append(item[1].numpy())
        error.append(np.sqrt(np.mean((Y_Test-Y_hat1)**2)))
        print('hi')
        print('-----------------------------------------')
        print('-------------My error Test-------------------')
        print(error)
        
        #Train
        Y_hat2=mdl.predict(Data_Train)
        Y_Train=[]
        for item in Data_Train:
            Y_Train.append(item[1].numpy())
        error_train=(np.sqrt(np.mean((Y_Train-Y_hat2)**2)))
        print('------------------------------------------')
        print('------------------My error Train----------')
        print(error_train)

        fig=plt.pyplot.figure(figsize=(12,6))
        ax=fig.add_subplot(1,3,1)
        ax.plot(hist['loss'])
        ax.set_title('Train loss')
        ax=fig.add_subplot(1,3,2)
        ax.plot(hist['root_mean_squared_error'])
        ax.set_title('RMSE Train')
        ax=fig.add_subplot(1,3,3)
        Y_Test=np.array(Y_Test)
        Y_Test=np.reshape(Y_Test,np.shape(Y_hat1))
        ax.plot(Y_hat1,Y_Test, 'o')
        x=range(-4,8)
        plt.pyplot.plot(x,'o:r')
        plt.pyplot.show()
        
        pdb.set_trace()
        
#---------------------------------------------------  
#---------------------------------------------------
# Step1: read data
path='E:\\PhD_thesis\\research\\semisuppervised\\drug_target\\thesis_code\\negative_examples\\checked_codes\\upload\\Code\\paper_2_code\\upload\\'
f_name=path+'Data\\Davis\\Davis.csv'
Prot_name, Drug_id, KD_val=read_data(f_name)
# Step2:read similarities
f_name=path+'Data\\Davis\\Drug_Fingerprint_sim.csv'
All_D_id,Drug_S=read_simarity(f_name)
f_name=path+'Data\\Davis\\Davis_Prot_SW_sim.csv'
All_P_id,Prot_S=read_simarity(f_name)
# Step3:create dataset   
Data_set,X,Y=create_dataset(Prot_name, Drug_id, KD_val,All_D_id,Drug_S,All_P_id,Prot_S)
# Step4: create model
#parameters
num_filters=8
kernel_size=5
pool_size=2
num_hidden_layer=1
num_hidden_unit=64
num_Con_layer=1
num_features=np.shape(X)[1]
mdl=NN_model(num_features,num_Con_layer,num_filters,kernel_size,pool_size,
         num_hidden_layer,num_hidden_unit)
# Step 5: Cross validation
num_fold=5
mode='Warm'
CV_evaluation(Data_set,Y,mdl,num_fold,mode)


