# mlp for the two circles classification problem
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,BatchNormalization,Dropout,GaussianNoise,InputLayer,Input
from keras.callbacks.callbacks import ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.optimizers import SGD,Adam
from keras.initializers import RandomUniform
from matplotlib import pyplot
import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../modules')
from simplednn import load_data,preprocess
from utils import *
from modules import *
import numpy as np
import keras

(x, y), (x_val, y_val)=load_data()

opt=SGD(learning_rate=0.01,momentum=0.9)
loss='binary_crossentropy'
dataset={}
dataset['x']=x
dataset['y']=y
dataset['x_val']=x_val
dataset['y_val']=y_val
epoch=100#1000
batch_size=128
log_dir='../../log/case20_tmp/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path=log_dir+'case20_tmp.csv'
fig_name=log_dir+'case20_tmp.pdf'
#reducelr=ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, min_lr=0.001)
#callbacks=[reducelr]#reducelr
callbacks=[]
model_path='/data/zxy/DL_tools/DL_tools/models/TrainingUnstable/case20_tmp.h5'

config={}
config['opt']=opt
config['loss']=loss
config['dataset']=dataset
config['epoch']=epoch
config['batch_size']=batch_size
config['callbacks']=callbacks
'''
# define model
model = Sequential()
#init = RandomUniform(minval=0, maxval=1)
init = 'he_uniform'

model.add(InputLayer(batch_input_shape=(None,2)))
#model.add(GaussianNoise(stddev=0.1))
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128,activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128,activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128,activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(128,activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))'''
init='he_uniform'
input = Input(shape=(2,))
x=Dense(128,activation='relu', kernel_initializer=init)(input)
x=Dense(128, activation='relu', kernel_initializer=init)(x)
x=Dense(128, activation='relu', kernel_initializer=init)(x)
x=Dense(128, activation='relu', kernel_initializer=init)(x)
x=Dense(128, activation='relu', kernel_initializer=init)(x)
x=Dense(1,activation='sigmoid', kernel_initializer=init)(x)
model=Model(input, x)
'''
#2.增加更多层
model.add(Dense(5,activation='tanh', kernel_initializer=init))
model.add(Dropout(0.1))
model.add(Dense(5,activation='tanh', kernel_initializer=init))
model.add(Dropout(0.1))
model.add(Dense(5,activation='tanh', kernel_initializer=init))
model.add(Dropout(0.1))
model.add(Dense(5,activation='tanh', kernel_initializer=init))
model.add(Dropout(0.1))'''
#model.add(BatchNormalization())
#model.add(Dense(5,activation='tanh', kernel_initializer=init))
#model.add(BatchNormalization())
#model.add(Dense(5,activation='tanh', kernel_initializer=init))
#model.add(BatchNormalization())
#model.add(Dense(5,activation='tanh', kernel_initializer=init))
#model.add(BatchNormalization())
'''model.add(Dense(5,activation='tanh', kernel_initializer=init))
model.add(Dense(5,activation='tanh', kernel_initializer=init))

model.add(Dense(1, activation='sigmoid', kernel_initializer=init))'''

model.summary()
save_model(model,model_path)
trained_model,history,_=model_train(model=model,optimizer=opt,loss=loss,dataset=dataset,iters=epoch,batch_size=batch_size,log_path=log_path,callbacks=callbacks,verb=1)
issue_list=determine_issue(history,trained_model,threshold_low=1e-3,threshold_high=1e+3)
issue_list=['unstable']
rm=Repair_Module(model,config,issue_list)
result=rm.solve()
#result_dic=read_csv(log_path,epoch)
generate_fig(result_dic,fig_name)