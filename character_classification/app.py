

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

from PIL import Image
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as  K
import numpy as np
from skimage import transform,io
from skimage.filters import threshold_mean
import numpy as np
import flask
import io
import tensorflow as tf

graph = tf.get_default_graph()

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
obj = None

def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(32,32,1)))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(83))
    model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['accuracy'])
    return model

def read_image(image_path):
    # read
    im = io.imread(image_path)
        
    # resize
    im = transform.resize(im, (32,32), mode='symmetric', preserve_range=True)
    
    # threashold to convert it to binary
    thresh = threshold_mean(im)
    binary = im > thresh

    # binary conversion
    binary[binary == True] = 1
    binary[binary == False] = 0
    
    return binary

class Classifier(object):
    
    def __init__(self, path):
        
        self.model = get_model()
        
        self.model.load_weights(path)
        
        self.labels = ['=', '-', ',', '!', '(', ')', '[', ']', '{', '}', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'alpha', 'ascii_124', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z', '']
    
    def predict(self, img):
        img = np.array(img).reshape((1,32,32,1))
        global graph
        with graph.as_default():
            pred_y = self.model.predict(img, batch_size=32)
        pred_y = np.argmax(pred_y, axis=1)
        
        classes_y = []
        for i in range(len(pred_y)):
            classes_y.append(self.labels[pred_y[i]])
        
        return classes_y

def prepare_image(image):
    im = img_to_array(image)
    
    im = transform.resize(im, (32,32), mode='symmetric', preserve_range=True)
    
    # threashold to convert it to binary
    thresh = threshold_mean(im)
    binary = im > thresh

    # binary conversion
    binary[binary == True] = 1
    binary[binary == False] = 0
    
    return binary

def load_model():
    global obj
    obj = Classifier('./trained_models/model-epoch:16-acc:0.905-val_acc0.936.hdf5')


@app.route("/predict", methods=["POST"])
def predict():
    
    global obj
    
    data = {"success": False}
    
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            x = prepare_image(image)
            pred = obj.predict(prepare_image(image))
   
            data["predictions"] = pred[0]

            data["success"] = True

    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("Loading stuffs")
    load_model()
    app.run(host='0.0.0.0')
