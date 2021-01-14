import os
import numpy as np
import tensorflow as tf
import pandas as pd
import random, sys, random, pickle
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Input, LSTM, Dense, Dropout
from keras.regularizers import l2, l1
from keras.layers import Reshape, Activation, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import keras.backend as K
from tensorflow.python.client import device_lib

get_ipython().run_line_magic('matplotlib', 'inline')
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.35
session=tf.Session(config=config)

def create_timestep(dataset,n):
    dataX= []
    for i in range(len(dataset)-n+1):
        a = dataset[i:(i+n)]
        dataX.append(a)
    return np.array(dataX)
def count(data):  
    c0=0
    c1=0
    for i in range(len(data)):
        if np.argmax(data[i])==0:
            c0=c0+1
        elif np.argmax(data[i])==1:
            c1=c1+1
    return c0,c1
def label(data):
    k=[]
    for i in range(len(data)):
        if np.argmax(data[i])==0:
            k.append(0)
        elif np.argmax(data[i])==1:
            k.append(1)
    return np.array(k)

# data load
snp_raw=pd.read_excel("C:/Users/snp_data.xlsx",sheet_name='Sheet1',header=0)
#sse_raw=pd.read_excel("C:/Users/sse_data.xlsx",sheet_name='Sheet1',header=0)
#kos_raw=pd.read_excel("C:/Users/kos_data.xlsx",sheet_name='Sheet1',header=0)

# input data
snp_x=np.array(snp_raw.iloc[:,:-3])
#sse_x=np.array(sse_raw.iloc[:,:-3])
#kos_x=np.array(kos_raw.iloc[:,:-3])

# output data
snp_return,snp_direction=np.array(snp_raw['return']),np.array(snp_raw.iloc[:,-2:])
#sse_return,sse_direction=np.array(sse_raw['return']),np.array(sse_raw.iloc[:,-2:])
#kos_return,kos_direction=np.array(kos_raw['return']),np.array(kos_raw.iloc[:,-2:])


stock_lst=['snp']
#stock_lst=['snp','sse','kos']
sequence=[50]
#sequence=[5,10,20,30,50,100]
variable=[4] 
#variable=[4,9,13,22,43]
len_stock,len_variable,len_sequence=len(stock_lst),len(variable),len(sequence)
mod = sys.modules[__name__]

for i in range(len_stock): # snp, sse, kos
  for v in range(len_variable):
    for s in range(len_sequence):
        setattr(mod, 'lfs_return_{}_var{}_se{}'.format(stock_lst[i],variable[v],sequence[s]), [])
        setattr(mod, 'lfs_direction_{}_var{}_se{}'.format(stock_lst[i],variable[v],sequence[s]), [])
        setattr(mod, 'lfm_{}_var{}_se{}'.format(stock_lst[i],variable[v],sequence[s]), [])

        
drop_rate=0.5
initial_rate=0.001
step_epoch=50
lrList = []
def step_decay(epoch):
    lrate = initial_rate
    if epoch >= step_epoch:
        lrate = initial_rate*drop_rate
    elif epoch >=step_epoch*2:
        lrate = lrate*drop_rate**2
    elif epoch >=step_epoch*3:
        lrate = lrate*drop_rate**3
    elif epoch >=step_epoch*4:
        lrate = lrate*drop_rate**4
    lrList.append(lrate)
    return lrate


def lstm_forest_model(var,index,data,seq):
    lf_datax=np.stack([create_timestep(data[index[0],:],seq)],2)
    for i in range(var):
        if i==0:
            pass
        else:
            lf_datax=lf_datax([lf_datax,creat_timestep(data[index[i],:],seq)],2)
    return lf_datax
        

def test_LFS_direction(lf_datax,y_return,y_cross,var,seq,epoch):
   
    lis=range(len(index1))
    np.random.seed(seq+var+9)    
    index=random.sample(lis,var)  
    lf_datax=lstm_forest_model(var,index,data,seq)
    lr = LearningRateScheduler(step_decay)
    callbacks_list = [lr]

    # train, test data split
    print("total length: ",  len(lf_datax))
    section=int(lf_datax.shape[0]*0.8)

    y_cross=y_cross[seq-1:]
    train_x2, test_x, train_y_cross2,test_y_cross =lf_datax[:section,:,:], lf_datax[section:,:,:], y_cross[:section],y_cross[section:]
  
    #train sample random choice
    index_random2=random.sample(range(train_x2.shape[0]),int(train_x2.shape[0]*7/8))
    index_random=np.sort(index_random2)
    train_x=train_x2[index_random,:,:]
    train_y_cross=train_y_cross2[index_random]
    print("total train length : ",len(train_x2),"  /   ","bagging train length : ",len(train_x))
     
    # modeling
    K.clear_session() 
    
    main_input = Input(shape=(seq,var),name='main_input')
    m=LSTM(15,return_sequences=False,kernel_regularizer=l2(0.01))(main_input)
    
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dropout(0.3)(m)
    
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dropout(0.3)(m)
    main_cross=Dense(2,activation='softmax',name='main_cross')(m)

    model=Model(inputs=main_input,outputs=[main_cross])
    model.compile(optimizer='adam',loss={'main_cross':'categorical_crossentropy'},metrics=['accuracy'])
    history=model.fit({'main_input':train_x},{'main_cross':train_y_cross},shuffle=False,callbacks=callbacks_list,
                      validation_data=(test_x,[test_y_cross]),epochs=epoch,batch_size=256,verbose=0)
    # evaluation
    lfs_direction_predict=model.predict(test_x)
    print("test evaluate",model.evaluate(test_x,[test_y_cross]))

    return lfs_direction_predict

def test_LFS_return(data,y_return,y_cross,var,seq,epoch):

    lis=range(len(index1))
    np.random.seed(seq+var+9)    
    index=random.sample(lis,var)  
    lf_datax=lstm_forest_model(var,index,data,seq)

    lr = LearningRateScheduler(step_decay)
    callbacks_list = [lr]

    print("total length: ",  len(lf_datax))
    section=int(lf_datax.shape[0]*0.8)
    y_return = y_return[seq-1:]
    train_x2, test_x, train_y_return2,test_y_return =lf_datax[:section,:,:], lf_datax[section:,:,:],  y_return[:section],y_return[section:]
  
    #train sample random choice
    index_random2=random.sample(range(train_x2.shape[0]),int(train_x2.shape[0]*0.875))
    index_random=np.sort(index_random2)
    train_x=train_x2[index_random,:,:]
    train_y_return=train_y_return2[index_random]
    print("total train length : ",len(train_x2),"  /   ","bagging train length : ",len(train_x))
     
    # modeling
    K.clear_session() 

    main_input = Input(shape=(seq,var),name='main_input')
    m=LSTM(15,return_sequences=False)(main_input)

    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dropout(0.3)(m)
    
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dropout(0.3)(m)
 
    main_return=Dense(1,activation='linear',name='main_return')(m)
    
    model=Model(inputs=main_input,outputs=[main_return])
    model.compile(optimizer='adam',loss={'main_return':'mean_squared_error'})
    history=model.fit({'main_input':train_x},{'main_return':train_y_return},shuffle=False,callbacks=callbacks_list,
                      validation_data=(test_x,test_y_return),epochs=epoch,batch_size=256,verbose=0)
    
    #model evaluation
    lfm_return_predict=model.predict(test_x)
    print("test evaluate",model.evaluate(test_x,test_y_return))
    return lfm_return_predict

def test_LFM(data,y_return,y_cross,var,seq,epoch):

    lis=range(len(index1))
    np.random.seed(seq+var+9)    
    index=random.sample(lis,var)  
    lf_datax=lstm_forest_model(var,index,data,seq)
    lr = LearningRateScheduler(step_decay)
    callbacks_list = [lr]
    section=int(lf_datax.shape[0]*0.8)


    y_cross, y_return =y_cross[seq-1:], y_return[seq-1:]
    train_x2, test_x, train_y_cross2, train_y_return2,test_y_cross,test_y_return =lf_datax[:section,:,:], lf_datax[section:,:,:], y_cross[:section], y_return[:section],y_cross[section:],y_return[section:]
  
    #train sample random choice
    index_random2=random.sample(range(train_x2.shape[0]),int(train_x2.shape[0]*7/8))
    index_random=np.sort(index_random2)
    train_x=train_x2[index_random,:,:]
    train_y_cross=train_y_cross2[index_random]
    train_y_return=train_y_return2[index_random]

    # modeling
    K.clear_session() 
    main_input = Input(shape=(seq,var),name='main_input')
    m=LSTM(15,return_sequences=False)(main_input)
    
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    m=Dropout(0.3)(m)
    

    main_cross=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    main_cross=Dense(30,activation='relu',kernel_initializer='glorot_normal')(main_cross)
    main_cross=Dropout(0.3)(m)
    main_cross=Dense(2,activation='softmax',name='main_cross')(main_cross)

    main_return=Dense(30,activation='relu',kernel_initializer='glorot_normal')(m)
    main_return=Dense(30,activation='relu',kernel_initializer='glorot_normal')(main_return)
    main_return=Dropout(0.3)(main_return)
    main_return=Dense(1,activation='linear',name='main_return')(main_return)
    
    model=Model(inputs=main_input,outputs=[main_cross,main_return])
    model.compile(optimizer='adam',loss={'main_cross':'categorical_crossentropy','main_return':'mean_squared_error'},
                   metrics=['accuracy'],loss_weights={'main_cross':8,'main_return':1})
    print("model : ",model.summary())
    history=model.fit({'main_input':train_x},{'main_cross':train_y_cross,'main_return':train_y_return},shuffle=False,callbacks=callbacks_list,
                      validation_split=1/8,epochs=epoch,batch_size=256,verbose=0)
    
    # model evaluation
    lfm_predict=model.predict(test_x)
    print("test evaluate",model.evaluate(test_x,[test_y_cross,test_y_return]))
    return lfm_predict

    #If you want to verify the index, train prediction
    """
    lfm_train_predict=model.predict(train_x)
    return lfm_predict, index, lfm_train_predict
    """




# ensamble LSTM-Forest
epoch=300
for L in range(100): #number of the LSTM
    for v in range(len_variable): # number of variable 
        for s in range(len_sequence): # timestep
            print('snp  %s -th lstm forest, variable %s, sequence 50 ' % (L,variable[v]))
            getattr(mod,'lfm_snp_var{}_se{}'.format(variable[v],sequence[s])).append(test_LFM(snp_x,snp_y1,snp_y2,variable[v],sequence[s],epoch=epoch))
            getattr(mod,'lfs_direction_snp_var{}_se{}'.format(variable[v],sequence[s])).append(test_LFS_direction(snp_x,snp_y1,snp_y2,variable[v],sequence[s],epoch=epoch))
            getattr(mod,'lfs_return_snp_var{}_se{}'.format(variable[v],sequence[s])).append(test_LFS_return(snp_x,snp_y1,snp_y2,variable[v],sequence[s],epoch=epoch))




# save lstm forest 

#with open('lfm_sse_test.pkl', 'wb') as fout:                    
#with open('lfm_kos_test.pkl', 'wb') as fout:                    
with open('lfm_snp_test.pkl', 'wb') as fout:
  pickle.dump(Dict, fout)
