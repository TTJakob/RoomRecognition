# RoomRecognition
Recognition of Classrooms via Deeplearning 

## Getting Started und Idee

Die Ideen zu einer Indoor-Navigation kam mir während der Anfangszeit meines Studiums, als ich noch sehr oft auf der Suche nach Klassenzimmern und Räumen war. Vergleichbare Situationen erlebt man auch in Flüghäfen oder großen Firmenkomplexen.

Zugegebenermaßen ist die Idee von Indoor-Navigation nicht neu, eine genaue Indoor-Navigation über beispielsweise W-lan Ortung ist aktuell außerst aufwendig umzusetzen.

Das Gebäude in dem die Navigation erfolgen sollte ist das Gebäude meiner ehmaligen FH.

Natürlich ist es etwas ambitioniert direkt eine funktionierende Navigation zu erstellen, deswegen liegt der Fokus dieser arbeit auf der Erkennung der einzelnen Räume des Gebäudes. Die Räume sollten anhand von Fotos und mittels Deep Learning erkannt und vonander unterschieden werden.
In einem prototypischen Navigationssytem soll der User ein Foto von dem Raum in dem er sich befindet schießen. Der Raum und somit seine Position vom System erkannt werden und als Basis für eine Navigation dienen.

In meinen Forschungen ist mir aufgefallen, dass der Großteil der Forschungsprojekte auf die Erkennung von Objekten in einem Raum abziehlt und nicht auf den Raum an sich. Diese Tatsache bestärkte mich nochmal mich diesem Projekt zu widmen.
***
The ideas for indoor navigation came to me during the early days of my studies, when I was often looking for classrooms and rooms. Comparable situations are also experienced in airports or large company complexes.

Admittedly, the idea of indoor navigation is not new, but an accurate indoor navigation using for example W-lan positioning is currently extremely difficult to implement.

The building in which the navigation should take place is the building of my former university of applied sciences.

Of course it is a bit ambitious to create a working navigation directly, so the focus of this work is on the recognition of the individual rooms of the building. The rooms should be recognized and distinguished from each other by means of photos and deep learning.

In a prototypical navigation system the user should take a photo of the room he is in. The room and thus its position should be recognized by the system and serve as a basis for navigation.

In my research I noticed that the majority of research projects are aimed at the recognition of objects in a room and not at the room itself. This fact encouraged me to dedicate myself to this project again.


## Transfer learning

Transfer learning is used to build highly accurate computer vision models for your custom purposes, even when you have relatively little data. Transfer learning uses a pre-trained network and only trains a custom classifier that reduces development and training time and provides good results.

![Transfer Learning](https://tensorflow.rstudio.com/blog/images/keras-pretrained-convnet/swapping_fc_classifier.png)

A pre-trained network is simply a saved network previously trained on a large dataset, typically on a large-scale image classification task. If this original dataset is large enough and general enough, then the features learned by the pre-trained network can effectively act as a generic model of our visual world in therms of identifying shapes and objects and extract different features out of a picture. So one pre-trained model/convolutional base can prove useful for many different computer vision problems.

## Create a first neural network

To start off I used VGG16 as a pretrained network. It´s a "simpel" neural network that is easy to implement and it´s and its architecture offers solid results proven in various international challenges.

I have used VGG with the pretrained weights from the imagenet challenge and modified it to recognize the 14 different rooms which I took pictures of.

The VGG Model is pretrained but not customized for the task of classifying rooms.

To optimze this Model for this specific usecase a new classifier is needed.
```

num_classes = 14


model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights=vgg16_weights_path,))
model.add(Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```

These lines initialise the VGG16 Model and add the new untrained classifier.

I have oriented myself towards the original classifier and used two dense layers with 1024 notes and a dropout layer inbetween.
The Outputlayer (last) sorts the picture into one of the 14 classes.

```
# Say not to train first layer (VGG16) model. It is already trained
model.layers[0].trainable = False
```
The VGG 16 is allready trained so we dont have to train hit again (this time).

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
VGG16 with 0 trainable parameters.
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
The custom classifier with roughtly 1.6 Mio prameters to train.

This is a massive improvement in training time compared to the 15 Mio parameters of the VGG 16 Model.

Link to the original Paper of the VGG architecture [VGG Model](https://arxiv.org/abs/1409.1556)6)


```
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
```
Initializing the optimizer and compiling the model.
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
```
To artificially "increase" the number of training pictures we use data augmentation. It intruduces a variaty of options to manipulate a image before it is loaded into the neural network.
```
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
With these generatores the batch size can be adjusted and the directory of the pictures is referenced.
```
history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,
      epochs=75,
      validation_data=validation_generator,
      validation_steps=5,
      )
```
The model.fit_generator defines the training variables and initializes the training.

## Training Results

<p align="center">
<img src="https://github.com/TTJakob/RoomRecognition/blob/pictures/VGG16_004_Aug_conv.PNG"   /> 
<p>

With a Training accuracy of just around 90% and a validation accuracy of rougthly 80% its a good first shot.
The validation loss is about 0,8.
These results are ok for an early network but can be improved using varius techniques.

## Finetuning

Finetuning is a technique in transfer learning that adapts the network even more to the task at hand and thus improves the results.

In a further training phase, parts of the previously frozen network will also be included in the training.

Finetuning of the network should start with the last layers to avoid too big changes in the network. Changes to early layers of networks can result in large variations and bad results.
```
model.layers[0].trainable = True

set_trainable = False
for layer in model.layers[0].layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```
In this section the last block of the VGG16 architecture is unforzen and therefore trained again.


```
model.layers[0].summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, None, None, 3)     0         
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
global_average_pooling2d_2 ( (None, 512)               0         
=================================================================
Total params: 14,714,688
Trainable params: 7,079,424
Non-trainable params: 7,635,264
_________________________________________________________________
```
The last block is unforzen so we have to train its parameters again.
```
model.summary()
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 512)               14714688  
_________________________________________________________________
dense_4 (Dense)              (None, 1024)              525312    
_________________________________________________________________
dropout_2 (Dropout)          (None, 1024)              0         
_________________________________________________________________
dense_5 (Dense)              (None, 1024)              1049600   
_________________________________________________________________
dense_6 (Dense)              (None, 14)                14350     
=================================================================
Total params: 16,303,950
Trainable params: 8,668,686
Non-trainable params: 7,635,264
_________________________________________________________________
```
Now we have over 8.6 Mio Paramters to train.
```
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
```
The optimizer is initalized with a lower learningrate (lr) to avoid large changes on a functioning networks which can lead to way worse results.

```
train_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input,
#train_datagen = ImageDataGenerator(
       rotation_range=30,
     width_shift_range=0.2,
     height_shift_range=0.2,
     shear_range=0.2,
      zoom_range=0.2,
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
Data Augmention is utilized again and the genarators remain vastly unchanged.
```
history = model.fit_generator(
      train_generator,
      steps_per_epoch=10,
      epochs=75,
      validation_data=validation_generator,
      validation_steps=5,
      )
      
      Epoch 74/75
10/10 [==============================] - 5s 530ms/step - loss: 0.0935 - acc: 0.9550 - val_loss: 0.3419 - val_acc: 0.9100
Epoch 75/75
10/10 [==============================] - 5s 539ms/step - loss: 0.1427 - acc: 0.9500 - val_loss: 0.5414 - val_acc: 0.8800


```
With the model.fit_generator the Network can be trained again.

## Finetuning Results

<p align="center">
<img src="https://github.com/TTJakob/RoomRecognition/blob/pictures/FTVGG16_004_Aug_Adam_conv_Layer4plus5.PNG"   /> 
<p>

With the finetuning the results could be improved again. The training accuracy could be increased to 95 and the validation accuracy to about 90%.

The loss of validation could also be reduced to below 0.5, which is also a positive development and can be used as a base model for an indor navigaiton prototype.

## Future Work
Inception V3 and Resnet50 TBA
