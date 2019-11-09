#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.datasets import mnist
import pandas as pd
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[ ]:


import matplotlib.pyplot as plt
i = 1
plt.imshow(train_images[i],cmap=plt.cm.binary)
plt.show()
print(train_labels[i])


# In[ ]:


train_images = train_images.reshape((60000,28,28,1))
test_images = test_images.reshape((10000,28,28,1))

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
                                                           


# In[ ]:


from keras import layers
from keras import models

network = models.Sequential()
network.add(layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation="relu"))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(64, (3,3), activation="relu"))
network.add(layers.MaxPooling2D((2,2)))

network.add(layers.Conv2D(64, (3,3), activation="relu"))
network.add(layers.MaxPooling2D((2,2)))

#network.add(layers.Conv2D(128,(3,3),activation="relu"))
network.add(layers.Flatten())
network.add(layers.Dense(units=32,activation="relu"))
network.add(layers.Dense(units=10,activation="softmax"))


# In[ ]:


network.summary()


# In[ ]:


network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


history = network.fit(train_images,train_labels,epochs=10,shuffle=True,batch_size=128)


# In[ ]:


test_loss,test_acc = network.evaluate(test_images,test_labels)
print(test_loss,test_acc)


# In[ ]:


import matplotlib.pyplot as plt
plt.xlabel("Epcho")
plt.ylabel("Loss")
plt.plot(history.history['loss'])


# In[ ]:


network.save('ccn.h5')


# In[1]:


import cv2
import numpy as np
from keras.models import load_model
#from model import NeuralNet

#net = NeuralNet()

model = load_model('ccn.h5')
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
        
        g = image.reshape(1,28,28,1)
        dimension = g.shape
        print(dimension)
        
        result = model.predict(g)
        #result = 0
        print("PREDICTION : ",result)
        index = result.argmax()
        print("VALUE :",label[index])

cv2.destroyAllWindows()  

