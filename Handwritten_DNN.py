#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://medium.com/@edwardpie/building-a-cnn-for-recognising-mouse-drawn-digits-with-keras-opencv-mnist-72a7ae7a070a
#https://github.com/hackstock/deep-ocr/blob/master/app.py
#https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/
from keras.datasets import mnist
import pandas as pd
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[2]:


train_images.shape


# In[3]:


import matplotlib.pyplot as plt
plt.imshow(train_images[0], cmap="gray")
plt.show()


# In[4]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[5]:


train_labels[:5]


# In[6]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[7]:


train_labels[:5]


# In[17]:


from keras import layers
from keras import models

network = models.Sequential()
network.add(layers.Dense(units=512,activation="relu",input_shape=(28*28,)))
network.add(layers.Dense(units=512,activation="relu"))
network.add(layers.Dense(units=10,activation="softmax"))


# In[18]:


network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])


# In[19]:


history = network.fit(train_images,train_labels,epochs=6,shuffle=True,batch_size=128)


# In[20]:


test_loss,test_acc = network.evaluate(test_images,test_labels)
print(test_loss,test_acc)


# In[12]:


import matplotlib.pyplot as plt
plt.xlabel("Epcho")
plt.ylabel("Loss")
plt.plot(history.history['loss'])


# In[ ]:


import cv2
import numpy as np
#from model import NeuralNet

#net = NeuralNet()

# creating a 600 x 600 pixels canvas for mouse drawing
canvas = np.ones((150,150), dtype="uint8") * 255
# designating a 400 x 400 pixels point of interest on which digits will be drawn
canvas[:,:] = 0

start_point = None
end_point = None
is_drawing = False

label = np.arange(0,10)

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,15)

def on_mouse_events(event,x,y,flags,params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x,y)
            draw_line(canvas,start_point,end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


cv2.namedWindow("Test Canvas")
cv2.setMouseCallback("Test Canvas", on_mouse_events)


while(True):
    cv2.imshow("Test Canvas", canvas)
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = True
    elif key == ord('c'):
        canvas[:,:] = 0
    elif key == ord('p'):
        image = canvas[:,:]
        dimension = image.shape
        print(dimension)
        
        #image = cv2.resize(image,(28,28)).flatten()
        image = cv2.resize(image,(28,28))
        image = image.astype('float32') / 255
        
        dimension = image.shape
        print(dimension)
        
        g = image.reshape(1,(28*28))
        dimension = g.shape
        print(dimension)
        
        result = network.predict(g)
        #result = 0
        print("PREDICTION : ",result)
        index = result.argmax()
        print("VALUE :",label[index])

cv2.destroyAllWindows()  

