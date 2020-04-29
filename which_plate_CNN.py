##My cnn, classyfing the plates as dirty, clean or full.
#imports
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing:
classifier = Sequential()

#Convolution:
classifier.add(Convolution2D(32, (3, 3), input_shape =(256, 256, 3), activation = "relu"))

#Pooling:
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#Flattening:
classifier.add(Flatten())

#Fully connected layers::
classifier.add(Dense(units = 128, activation = "relu"))
classifier.add(Dense(units = 3, activation = "softmax"))

#Making CNN:
##If there are more than 2 possible outcomes, the loss is "categorical_crossentropy";
classifier.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

#classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

#From KERAS:
from keras.preprocessing.image import ImageDataGenerator

#Data augmentation:
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.1,
        fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('plates/training_set',
										         target_size=(256, 256),
										         batch_size=16,
										         class_mode='categorical')

test_set = test_datagen.flow_from_directory('plates/test_set',
											 target_size=(256, 256),
											 batch_size=32,
											 class_mode='categorical')

classifier.fit_generator(
			        training_set,
			        epochs=50,
			        validation_data=test_set,
			        validation_steps=2000)

#accuracy: batch size 16, epochs 25 == 88%