'''对比实验 共享前7个block，分别拥有1层'''
# -*- coding: utf-8 -*-
import keras
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from keras.engine import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import regularizers, initializers
import numpy as  np
import pandas as pd
import os

#config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
#sess = tf.compat.v1.Session(config=config)

##############将内容输出到txt
import sys

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('/data/houxiaoyuan/face_multi-task_own/results/749_control_7_1/749_control_7_1.txt')  # 调用print时相当于Logger().write()

##########################


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


filename = '749_control_7_1'


# 新加一个用于画出图 用tensorboard来画
# keras.callbacks.TensorBoard(log_dir='/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/graph',
#                             histogram_freq=0,
#                             write_graph=True,
#                             write_images=True)
# tbCallBack = keras.callbacks.TensorBoard(log_dir='/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/graph',
#                                          histogram_freq=0,
#                                          write_graph=True,
#                                          write_images=True)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# custom parameters
nb_class1 = 7          #表情
nb_class2 = 107        #身份
nb_class3 = 2          #性别
hidden_dim = 512
img_w = 224
img_h = 224
batchsize = 32
epochs = 500
dropout_rate = 0.5
weight1 = 0.4
weight2 = 0.3
weight3 = 0.3

vgg_model = VGGFace(include_top=True, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('flatten').output

x = Dense(hidden_dim, activation='relu', name='fc6', kernel_regularizer=regularizers.l2(0.001))(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc7', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(dropout_rate)(x)


out1 = Dense(nb_class1, activation='softmax', name='emo_fc8', kernel_regularizer=regularizers.l2(0.001))(x)
# out1表情

out2 = Dense(nb_class2, activation='softmax', name='id_fc8',
             kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',
                                                             seed=None))(x)
#out2 身份

out3 = Dense(nb_class3, activation='softmax', name='gen_fc8',
             kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None))(x)
# out3性别

custom_vgg_model = Model(vgg_model.input, [out1, out2,out3])

# #从conv5_1开始训，15对应的是pool4
for layer in custom_vgg_model.layers[:20]:
    layer.trainable = False # 把前面的层的权重设置为不训练
for layer in custom_vgg_model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)

custom_vgg_model.summary()
sgd = SGD(lr=0.001)  # 设置优化器
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto')

custom_vgg_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                         optimizer=sgd,
                         loss_weights=[weight1, weight2, weight3],
                         metrics=['accuracy'])

print('model is ready')

#this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_frame = pd.read_excel('/data/houxiaoyuan/face_multi-task_own/data/749_train_new.xls')


def encode(data_frame, col):
    col_lst = list(set(data_frame[col].values))
    col_value = [1 * (np.array(col_lst) == value) for value in data_frame[col]]
    data_frame[col] = col_value
    return data_frame

for data_col in ['emo', 'id','gender']:
    train_frame = encode(train_frame, data_col)


# test_frame 相同
train_generator = train_datagen.flow_from_dataframe(
    train_frame,  # train_frame this is the target directory
    directory='/data/houxiaoyuan/face_multi-task_own/data/KDEF_NOKID/749_train_new/',
    x_col='path',  # 文件路径
    y_col=['emo', 'id','gender'],  # 'emo' 表情列名�?'id' 序号列名
    target_size=(img_w, img_h),
    batch_size=batchsize,
    class_mode='multi_output')

N_train = train_generator.n
print("N_train:", N_train)

# this is a similar generator, for validation data
print('generator is ok,ready to train')

checkpointer = ModelCheckpoint('/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/model/{epoch:03d}.hdf5',   #每个epoch后保存模型
								verbose=0,
								save_best_only=False,
								save_weights_only=True,
								period=1)

history1=custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=N_train / batchsize,
    epochs=epochs,
    #callbacks=[tbCallBack,reduce_lr,early_stopping])
    # callbacks=[tbCallBack,checkpointer])
    callbacks=[checkpointer])
accemo=history1.history['emo_fc8_acc']
accid=history1.history['id_fc8_acc']
accgen=history1.history['gen_fc8_acc']

loss = history1.history['loss']
lossemo=history1.history['emo_fc8_loss']
lossid=history1.history['id_fc8_loss']
lossgen=history1.history['gen_fc8_loss']

np_accemo = np.array(accemo).reshape((1,len(accemo))) #reshape是为了能够跟别的信息组成矩阵一起存�?
np_accid = np.array(accid).reshape((1,len(accid)))
np_accgen = np.array(accgen).reshape((1,len(accgen)))

np_loss =np.array(loss).reshape(1,len(loss))
np_lossemo=np.array(lossemo).reshape(1,len(lossemo))
np_lossid=np.array(lossid).reshape(1,len(lossid))
np_lossgen=np.array(lossid).reshape(1,len(lossgen))

np_outacc = np.concatenate([np_accemo,np_accid,np_accgen],axis=0)
np_outloss = np.concatenate([np_loss,np_lossemo,np_lossid,np_lossgen],axis=0)

np.savetxt('/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/saveacc-0505.txt',np_outacc)
np.savetxt('/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/saveloss-0505.txt',np_outloss)
print("save successfully")
print('training is over')

json_string = custom_vgg_model.to_json()
open('/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/'+filename+'.json', 'w').write(json_string)  # 保存结构
custom_vgg_model.save_weights('/data/houxiaoyuan/face_multi-task_own/results/'+filename+'/'+filename+'.h5')  # 保存权重
print('save model successfully')