# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from keras_vggface.vggface import VGGFace
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers, initializers
import numpy as  np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


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
sys.stdout = Logger('xxx/xxx/xxx/xxxx.txt')  # logger address


# custom parameters
all_class = 2          #Number of categories

hidden_dim = 512
img_w = 224
img_h = 224
batchsize = 32
epochs = 500
dropout_rate = 0.5


vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
# vgg_model.summary()
last_layer = vgg_model.get_layer('pool5').output

x = Flatten(name='flatten1')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dense(hidden_dim, activation='relu', name='fc7', kernel_regularizer=regularizers.l2(0.001))(x)
x = Dropout(dropout_rate)(x)
out = Dense(all_class, activation='softmax', name='fc8',kernel_regularizer=regularizers.l2(0.001))(x)

custom_vgg_model = Model(vgg_model.input, out)

custom_vgg_model.summary()
sgd = SGD(lr=1e-8)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, mode='auto')

custom_vgg_model.compile(loss='categorical_crossentropy',
                         optimizer=sgd,
                         metrics=['accuracy'])

print('model is ready')



train_datagen = ImageDataGenerator(
    samplewise_center=True,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


train_frame = pd.read_excel('xxx/xxx/xxx/xxxx.xls',dtype='str') #Table of data labels



train_generator = train_datagen.flow_from_dataframe(
    train_frame,
    directory='xxx/xxx/xxx/xxxx', #data address
    x_col='path',
    y_col='gen',
    target_size=(img_w, img_h),
    batch_size=batchsize
)

N_train = train_generator.n
print("N_train:", N_train)

# this is a similar generator, for validation data
print('generator is ok,ready to train')

checkpointer = ModelCheckpoint('/xxx/xxx/xxx/xxx.hdf5',  # model save address
                               verbose=0,
                               save_best_only=False,
                               save_weights_only=True,
                               period=1)

history1=custom_vgg_model.fit_generator(
    train_generator,
    steps_per_epoch=N_train / batchsize,
    epochs=epochs,
    callbacks=[checkpointer])
acc=history1.history['acc']

loss = history1.history['loss']

np.savetxt('/xxx/xxx/xxx/xxx.txt', acc) #acc save address
np.savetxt('/xxx/xxx/xxx/xxx.txt', loss)#loss save address
print("save successfully")
print('training is over')

json_string = custom_vgg_model.to_json()
open('/xxx/xxx/xxx/xxx.json', 'w').write(json_string)  # network structure save address
custom_vgg_model.save_weights('/xxx/xxx/xxx/xxx.h5')  #network weight save address
print('save model successfully')
