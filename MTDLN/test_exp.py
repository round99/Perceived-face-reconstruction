import numpy as np
import os
import cv2
from keras.models import model_from_json
from keras.engine import Model
from keras.utils.np_utils import to_categorical
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def string_to_float(str):
    return float(str)


def evaluate(weight, model):
    model.load_weights(weight)
    model.summary()


    fc8_layer = Model(inputs=model.input, outputs=model.get_layer(name='fc8').output)

    correct = 0
    imgs_num = 140  # Number of test images


    for root, dirs, files in os.walk('/xxx/xxx/xxx/'): #test dataset address

        for d in dirs:
            dirpath = os.path.join(root, d)
            imgs = os.listdir(dirpath)
            imgnum = len(imgs)

            for j in range(imgnum):
                imgpath = os.path.join(dirpath, imgs[j])
                raw_img = cv2.imread(imgpath)
                test_img = cv2.resize(raw_img, (224, 224))
                test_img = np.array(test_img)
                test_img = (test_img.astype(np.float32)) / 255.0
                test_img = np.expand_dims(test_img, axis=0)
                softmax_res = fc8_layer.predict(test_img, batch_size=64)
                max_index = np.argmax(softmax_res)
                if int(d) == max_index:
                    correct += 1


    acc = correct / imgs_num

    return acc


if __name__ == '__main__':

    models_path = '/xxx/xxx/xxx/'  # Model weight address
    acc = []

    model = model_from_json(open('/xxx/xxx/xxx.json','r').read())  # Model structure address
    files = [os.path.join(models_path, f) for f in os.listdir(models_path)]
    files.sort()
    for weight_path in files:
        acc1 = evaluate(weight_path, model)
        acc.append(acc1)
        np.save('/xxx/xxx/xxx/acc.npy', acc)


