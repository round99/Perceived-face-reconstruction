# -*- coding: utf-8 -*-
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from tensorflow.keras.models import Model
from keras_vggface.vggface import VGGFace
from tensorflow.keras.optimizers import SGD
from keras_preprocessing_image.image_data_generator import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers, initializers
import numpy as np
import pandas as pd
import os
import sys

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        self.log = open(fileN, 'a')

    def write(self, message):

        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('xxx/xxx/xxx/xxxx.txt')  # logger address


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


tbCallBack = keras.callbacks.TensorBoard(log_dir='xxx/xxx/xxx/xxx',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# custom parameters
nb_class1 = 7          #Number of Expression Categories
nb_class2 = 107        #Number of Identity Categories
nb_class3 = 2          #Number of Gender Categories
hidden_dim = 512
img_w = 224
img_h = 224
batchsize = 32
epochs = 500
dropout_rate = 0.5
weight1 = 0.4
weight2 = 0.3
weight3 = 0.3

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool3').output

x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv4_1')(last_layer)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv4_2')(x1)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv4_3')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='emo_pool4')(x1)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_1')(x1)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_2')(x1)
x1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='emo_conv5_3')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='emo_pool5')(x1)
x1 = Flatten(name='emo_flatten')(x1)
x1 = Dense(hidden_dim, activation='relu', name='emo_fc6', kernel_regularizer=regularizers.l2(0.001))(x1)
x1 = Dense(hidden_dim, activation='relu', name='emo_fc7', kernel_regularizer=regularizers.l2(0.001))(x1)
x1 = Dropout(dropout_rate)(x1)
out1 = Dense(nb_class1, activation='softmax', name='emo_fc8', kernel_regularizer=regularizers.l2(0.001))(x1)
# out1 Expression

x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv4_1')(last_layer)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv4_2')(x2)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv4_3')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='id_pool4')(x2)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_1')(x2)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_2')(x2)
x2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='id_conv5_3')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='id_pool5')(x2)
x2 = Flatten(name='id_flatten')(x2)
x2 = Dense(hidden_dim, activation='relu', name='id_fc6', kernel_regularizer=regularizers.l2(0.001))(x2)
x2 = Dense(hidden_dim, activation='relu', name='id_fc7', kernel_regularizer=regularizers.l2(0.001))(x2)
x2 = Dropout(dropout_rate)(x2)
out2 = Dense(nb_class2, activation='softmax', name='id_fc8',
             kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',
                                                             seed=None))(x2)
#out2 Identity


x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv4_1')(last_layer)
x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv4_2')(x3)
x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv4_3')(x3)
x3 = MaxPooling2D((2, 2), strides=(2, 2), name='gen_pool4')(x3)
x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv5_1')(x3)
x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv5_2')(x3)
x3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='gen_conv5_3')(x3)
x3 = MaxPooling2D((2, 2), strides=(2, 2), name='gen_pool5')(x3)
x3 = Flatten(name='gen_flatten')(x3)
x3 = Dense(hidden_dim, activation='relu', name='gen_fc6', kernel_regularizer=regularizers.l2(0.001))(x3)
x3 = Dense(hidden_dim, activation='relu', name='gen_fc7', kernel_regularizer=regularizers.l2(0.001))(x3)
x3 = Dropout(dropout_rate)(x3)
out3 = Dense(nb_class3, activation='softmax', name='gen_fc8',
             kernel_initializer=initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal',seed=None))(x3)
# out3 Gender

custom_vgg_model = Model(vgg_model.input, [out1, out2, out3])


for layer in custom_vgg_model.layers[:11]:
    layer.trainable = False
for layer in custom_vgg_model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)

custom_vgg_model.summary()
sgd = SGD(lr=0.001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto')

custom_vgg_model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy','categorical_crossentropy'],
                         optimizer=sgd,
                         loss_weights=[weight1, weight2, weight3],
                         metrics=['accuracy'])

print('model is ready')


train_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_frame = pd.read_excel('xxx/xxx/xxx/xxxx.xls',dtype='str') #Table of data labels


def encode(data_frame, col):
    col_lst = list(set(data_frame[col].values))
    col_value = [1 * (np.array(col_lst) == value) for value in data_frame[col]]
    data_frame[col] = col_value
    return data_frame

for data_col in ['emo', 'id','gender']:
    train_frame = encode(train_frame, data_col)



train_generator = train_datagen.flow_from_dataframe(
    train_frame,
    directory='xxx/xxx/xxx/xxxx', #data address
    x_col='path',
    y_col=['emo', 'id','gender'],
    target_size=(img_w, img_h),
    batch_size=batchsize,
    class_mode='multi_output')

N_train = train_generator.n
print("N_train:", N_train)
print('generator is ok,ready to train')

checkpointer = ModelCheckpoint('/xxx/xxx/xxx/xxx.hdf5',   #model save address
								verbose=0,
								save_best_only=False,
								save_weights_only=True,
								period=1)

history1=custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=N_train / batchsize,
    epochs=epochs,
    callbacks=[tbCallBack, checkpointer])
accemo=history1.history['emo_fc8_acc']
accid=history1.history['id_fc8_acc']
accgen=history1.history['gen_fc8_acc']

loss = history1.history['loss']
lossemo=history1.history['emo_fc8_loss']
lossid=history1.history['id_fc8_loss']
lossgen=history1.history['gen_fc8_loss']

np_accemo = np.array(accemo).reshape((1,len(accemo)))
np_accid = np.array(accid).reshape((1,len(accid)))
np_accgen = np.array(accgen).reshape((1,len(accgen)))

np_loss =np.array(loss).reshape(1,len(loss))
np_lossemo=np.array(lossemo).reshape(1,len(lossemo))
np_lossid=np.array(lossid).reshape(1,len(lossid))
np_lossgen=np.array(lossid).reshape(1,len(lossgen))

np_outacc = np.concatenate([np_accemo,np_accid,np_accgen],axis=0)
np_outloss = np.concatenate([np_loss,np_lossemo,np_lossid,np_lossgen],axis=0)

np.savetxt('/xxx/xxx/xxx/xxx.txt', np_outacc) #acc save address
np.savetxt('/xxx/xxx/xxx/xxx.txt', np_outloss)#loss save address
print("save successfully")
print('training is over')

json_string = custom_vgg_model.to_json()
open('/xxx/xxx/xxx/xxx.json', 'w').write(json_string)  # network structure save address
custom_vgg_model.save_weights('/xxx/xxx/xxx/xxx.h5')  #network weight save address
print('save model successfully')