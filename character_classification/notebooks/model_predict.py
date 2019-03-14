{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ[\"CUDA_DEVICE_ORDER\"]=\"PCI_BUS_ID\"   # see issue #152\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.models import Model\n",
    "from keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D\n",
    "from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img\n",
    "from keras.callbacks import ModelCheckpoint, TensorBoard\n",
    "from keras import backend as  K\n",
    "import numpy as np\n",
    "from skimage import transform,io\n",
    "from skimage.filters import threshold_mean\n",
    "import numpy as np\n",
    "import flask\n",
    "import io"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# initialize our Flask application and the Keras model\n",
    "app = flask.Flask(__name__)\n",
    "obj = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_model():\n",
    "    model = Sequential()\n",
    "    model.add(Conv2D(32, (3, 3), padding='same',\n",
    "                     input_shape=(32,32,1)))\n",
    "    model.add(Activation('relu'))\n",
    "\n",
    "    model.add(Conv2D(32, (3, 3)))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    model.add(Dropout(0.25))\n",
    "\n",
    "    model.add(Conv2D(64, (3, 3), padding='same'))\n",
    "    model.add(Activation('relu'))\n",
    "\n",
    "    model.add(Conv2D(64, (3, 3)))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(MaxPooling2D(pool_size=(2, 2)))\n",
    "    model.add(Dropout(0.25))\n",
    "\n",
    "    model.add(Flatten())\n",
    "\n",
    "    model.add(Dense(512))\n",
    "    model.add(Activation('relu'))\n",
    "    model.add(Dropout(0.5))\n",
    "\n",
    "    model.add(Dense(83))\n",
    "    model.add(Activation('softmax'))\n",
    "\n",
    "#     model.compile(loss='categorical_crossentropy',\n",
    "#                   optimizer='adam',\n",
    "#                   metrics=['accuracy'])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def read_image(image_path):\n",
    "    # read\n",
    "    im = io.imread(image_path)\n",
    "        \n",
    "    # resize\n",
    "    im = transform.resize(im, (32,32), mode='symmetric', preserve_range=True)\n",
    "    \n",
    "    # threashold to convert it to binary\n",
    "    thresh = threshold_mean(im)\n",
    "    binary = im > thresh\n",
    "\n",
    "    # binary conversion\n",
    "    binary[binary == True] = 1\n",
    "    binary[binary == False] = 0\n",
    "    \n",
    "    return binary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Classifier(object):\n",
    "    \n",
    "    def __init__(self, path):\n",
    "        \n",
    "        self.model = get_model()\n",
    "        \n",
    "        self.model.load_weights(path)\n",
    "        \n",
    "        self.labels = ['=', '-', ',', '!', '(', ')', '[', ']', '{', '}', '+', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'alpha', 'ascii_124', 'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash', 'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum', 'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z', '']\n",
    "    \n",
    "    def predict(self, img):\n",
    "        img = np.array(img).reshape((1,32,32,1))\n",
    "        pred_y = self.model.predict(img, batch_size=32)\n",
    "        pred_y = np.argmax(pred_y, axis=1)\n",
    "        \n",
    "        classes_y = []\n",
    "        for i in range(len(pred_y)):\n",
    "            classes_y.append(self.labels[pred_y[i]])\n",
    "        \n",
    "        return classes_y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/aditya16217/.local/lib/python3.6/site-packages/skimage/transform/_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.\n",
      "  warn(\"Anti-aliasing will be enabled by default in skimage 0.15 to \"\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "['1']"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def prepare_image(image):\n",
    "    im = img_to_array(image)\n",
    "    \n",
    "    im = transform.resize(im, (32,32), mode='symmetric', preserve_range=True)\n",
    "    \n",
    "    # threashold to convert it to binary\n",
    "    thresh = threshold_mean(im)\n",
    "    binary = im > thresh\n",
    "\n",
    "    # binary conversion\n",
    "    binary[binary == True] = 1\n",
    "    binary[binary == False] = 0\n",
    "    \n",
    "    return binary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_model():\n",
    "    global obj\n",
    "    obj = Classifier('../trained_models/model-epoch:16-acc:0.905-val_acc0.936.hdf5')v\n",
    "\n",
    "\n",
    "@app.route(\"/predict\", methods=[\"POST\"])\n",
    "def predict():\n",
    "    \n",
    "    global obj\n",
    "    \n",
    "    data = {\"success\": False}\n",
    "    \n",
    "    if flask.request.method == \"POST\":\n",
    "        if flask.request.files.get(\"image\"):\n",
    "            \n",
    "            # read the image in PIL format\n",
    "            image = flask.request.files[\"image\"].read()\n",
    "            image = Image.open(io.BytesIO(image))\n",
    "            \n",
    "            x = prepare_image(image)\n",
    "            pred = obj.predict(prepare_image(image))\n",
    "            \n",
    "            data[\"predictions\"] = pred[0]\n",
    "\n",
    "            data[\"success\"] = True\n",
    "\n",
    "    return flask.jsonify(data)\n",
    "\n",
    "# if this is the main thread of execution first load the model and\n",
    "# then start the server\n",
    "if __name__ == \"__main__\":\n",
    "    print(\"Loading stuffs\")\n",
    "    load_model()\n",
    "    app.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2\n"
     ]
    }
   ],
   "source": [
    "# import the necessary packages\n",
    "import requests\n",
    "\n",
    "# initialize the Keras REST API endpoint URL along with the input\n",
    "# image path\n",
    "KERAS_REST_API_URL = \"http://192.168.2.212:5000/predict\"\n",
    "\n",
    "IMAGE_PATH = '../data/extracted_images/2/2_100018.jpg'\n",
    "\n",
    "\n",
    "# load the input image and construct the payload for the request\n",
    "image = open(IMAGE_PATH, \"rb\").read()\n",
    "payload = {\"image\": image}\n",
    "\n",
    "# submit the request\n",
    "r = requests.post(KERAS_REST_API_URL, files=payload).json()\n",
    "\n",
    "# ensure the request was sucessful\n",
    "if r[\"success\"]:\n",
    "    print(r[\"predictions\"])\n",
    "else:\n",
    "    print(\"Request failed\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
