# RoomRecognition
Recognition of Classrooms via Deeplearning 
```
from keras import layers
from keras import models
from keras import optimizers
from keras import applications

import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
```
```
base_dir = 'E:/Bilder/Uni/BaRaume'
base_dir_models = 'E:/Bilder/Uni/TensModels'

train_dir = os.path.join(base_dir, 'trainMod')
validation_dir = os.path.join(base_dir, 'valMod')
test_dir = os.path.join(base_dir, 'test')

num_classes = 14

vgg16_weights_path = 'imagenet'

model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights=vgg16_weights_path,))
model.add(Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (VGG16) model. It is already trained
model.layers[0].trainable = False
```
rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9)
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
sgd =optimizers.SGD(lr=0.01, nesterov=False)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, None, None, 3)     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
_________________________________________________________________
global_average_pooling2d_1 ( (None, 512)               0         
=================================================================
Total params: 14,714,688
Trainable params: 0
Non-trainable params: 14,714,688
_________________________________________________________________
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 512)               14714688  
_________________________________________________________________
dense_1 (Dense)              (None, 1024)              525312    
_________________________________________________________________
dropout_1 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_3 (Dense)              (None, 14)                14350     
=================================================================
Total params: 16,303,950
Trainable params: 1,589,262
Non-trainable params: 14,714,688
```
```
train_datagen = ImageDataGenerator(
    preprocessing_function=applications.vgg16.preprocess_input,
    rescale=1./256,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
                                  )
                            

val_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        batch_size=20,
         class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        batch_size=20,
        class_mode='categorical')
```
```
history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,
      epochs=75,
      validation_data=validation_generator,
      validation_steps=5,
      )
```
<p align="center">
<img src="https://github.com/TTJakob/RoomRecognition/blob/pictures/VGG16_004_Aug_conv.PNG"  width="450" height="350" /> 
<p>

```
```
```
```
```
```
```
```
