# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 07:56:25 2022

@author: betsa
"""


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as plt
import csv
import pdb
import keras_tuner as kt
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
 fig=plt.pyplot.figure(figsize=(12,4))
 for i in range(0,N_Tr):
     ind1=ind_d_Tr[i]
     ind1=Sorted_drug_ind[ind1][-Win_sd:]
     ind2=ind_p_Tr[i]
     ind2=Sorted_prot_ind[ind2][-Win_sp:]
     t1=Y_matrix[ind1][:]
     temp_Y=t1[:,ind2]
     temp_Y=np.array(temp_Y)
     temp_Y1=temp_Y
     temp_Y1[-1][-1]=0
     '''avg=np.mean(temp_Y[np.nonzero(temp_Y)])
     index=np.argwhere(temp_Y==0)
     for item in index:
         temp_Y1[item[0]][item[1]]=avg'''

     X_Train.append(temp_Y1)  

     #---------   
 for i in range(0,N_Te):
    ind1=ind_d_Te[i]
    ind1=Sorted_drug_ind[ind1][-Win_sd:]
    ind2=ind_p_Te[i]
    ind2=Sorted_prot_ind[ind2][-Win_sp:]
    t1=Y_matrix[ind1][:]
    temp_Y=t1[:,ind2]
    temp_Y=np.array(temp_Y)
    '''avg=np.mean(temp_Y[np.nonzero(temp_Y)])
    index=np.argwhere(temp_Y==0)
    for item in index:
        temp_Y[item[0]][item[1]]=avg'''
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
    np.random.seed(8453)
    index=np.random.permutation(N)
    index=np.array(index)
    if mode=='Warm':
        low=np.int64((curr_fold-1)*fold_size)
        high=np.int64(curr_fold*fold_size)
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
        ind_d_Te=index_d[Test_ind]
        ind_p_Te=index_p[Test_ind]
        Y=np.array(Y)
        Y_Train=Y[Train_ind]
        Y_Test=Y[Test_ind]
        pdb.set_trace()
        
        
   
    
    
    return ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test  
#-------------------------------------------
class con_layer:
    def __init__(self,num_con,num_filter,kernel_size,win_size):
        self.num_con=num_con
        self.num_filter=num_filter
        self.kernel_size=kernel_size
        self.win_size=win_size
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
        XD=tf.keras.layers.Conv2D(filters=drug_info.num_filter,
                               kernel_size=drug_info.kernel_size,
                               activation='linear',padding='same',
                               strides=(1,1))(XD)
       
    XD=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XD)
    XD=tf.keras.layers.Flatten()(XD)
    # Prot
    for i in range(prot_info.num_con):
        XP=tf.keras.layers.Conv2D(filters=prot_info.num_filter,
                               kernel_size=drug_info.kernel_size,
                               activation='linear',padding='same',
                               strides=(1,1))(XP)
       
    XP=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XP)
    XP=tf.keras.layers.Flatten()(XP)
    # Interaction
    for i in range(interaction_info.num_con):
        XI=tf.keras.layers.Conv2D(filters=interaction_info.num_filter,
                               kernel_size=interaction_info.kernel_size,
                               activation='linear',padding='same',
                               strides=(1,1))(XI)
       
    XI=tf.keras.layers.MaxPooling2D(strides=(2,2),pool_size=(2, 2))(XI)
    XI=tf.keras.layers.Flatten()(XI)
    
    #--------------------------------------------------
    #concatenate
    X=tf.keras.layers.concatenate([XD,XP,XI])
    
    for i in range(0,num_hidden):
        X=tf.keras.layers.Dense(units=num_units,activation='linear')(X)
        X=tf.keras.layers.Dropout(rate=0.5)(X)
        num_units=np.floor(num_units/2)
    
    #-------------------------------------------------
    #output
    
    prediction=tf.keras.layers.Dense(units=1,activation=None)(X)
    #-------------------------------------------------
    #Model              
    mdl=tf.keras.Model(inputs=[Input_D,Input_P,Input_I],outputs=[prediction])                
    mdl.summary()
    mdl.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return mdl
#-------------------------------------------
def CV_evaluation(drug_info,prot_info,interaction_info,index_d,index_p,Y,Y_matrix ,mdl,num_fold,Drug_S,Prot_S):
    # parameters
    epochs=10
    batch_size=32
    folds=range(1,num_fold+1)
    error=[]
    for i in folds:
        ind_d_Tr,ind_p_Tr,ind_d_Te,ind_p_Te,Y_Train,Y_Test=cross_validation_Train_Test1(index_d,
                                                             index_p,Y,num_fold,i,'Warm')
        
        
        Input_D_Tr,Input_D_Te=create_sim_mat(drug_info.win_size,ind_d_Tr,ind_d_Te,Drug_S)
        Input_P_Tr,Input_P_Te=create_sim_mat(prot_info.win_size,ind_p_Tr,ind_p_Te,Prot_S)
        Input_I_Tr,Input_I_Te=create_Interaction_mat(interaction_info.win_size,ind_d_Tr,
                                                    ind_p_Tr,ind_d_Te,ind_p_Te,
                                                    Y_matrix,Y_Train,Y_Test,
                                                    Drug_S,Prot_S)
    
       
        Train_result=mdl.fit(([Input_D_Tr,Input_P_Tr,Input_I_Tr]),Y_Train,epochs=epochs,verbose=1)
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
       
        
        
        print('-------------Test Result--------------------')
        # Test
        Y_hat1=mdl.predict(([Input_D_Te,Input_P_Te,Input_I_Te])) 
        error.append(np.sqrt(np.mean((Y_Test-Y_hat1)**2)))
        print('---------------MY Test Error------------------')
        print(error)
        #Train
        Y_hat2=mdl.predict(([Input_D_Tr,Input_P_Tr,Input_I_Tr]))
        error_train=(np.sqrt(np.mean((Y_Train-Y_hat2)**2)))
        print('-----------------MT Train Error------------------')   
        print(error_train)

        Y_Test=np.array(Y_Test)
        Y_Test=np.reshape(Y_Test,np.shape(Y_hat1))
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
path='E:\\PhD_thesis\\research\\semisuppervised\\drug_target\\thesis_code\\negative_examples\\checked_codes\\upload\\Code\\paper_2_code\\upload\\'
f_name=path+'Data\\Davis\\Davis.csv'
Prot_name,Drug_id,KD_val=read_data(f_name) 
# 2-read similarities
f_name=path+'Data\\Davis\\Davis_Drug_Fingerprint_sim.csv'
All_D_id,Drug_S=read_simarity(f_name)
f_name=path+'Data\\Davis\\Davis_Prot_SW_sim.csv'
All_P_id,Prot_S=read_simarity(f_name)
# 3-create dataset   
index_d,index_p,Y,Y_matrix =create_dataset(Prot_name, Drug_id, KD_val,All_D_id,All_P_id)

# 4- create model
drug_info=con_layer(3,64,(5,5),(20,20))
prot_info=con_layer(3, 64, (5,5),(20,20))
interaction_info=con_layer(3, 64, (5,5),(20,20))
num_hidden=2
num_units=64
mdl=NN_model(drug_info,prot_info,interaction_info,num_hidden,num_units)
# 5- 5-CV evaluation
CV_evaluation(drug_info,prot_info,interaction_info,index_d,index_p,Y,Y_matrix ,mdl,5,Drug_S,Prot_S)
