import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.metrics import confusion_matrix  # Add this line
import seaborn as sns
from sklearn.metrics import classification_report


# Data paths
train_set = '../input/dataset/train'
val_set = '../input/dataset/val'
test_set = '../input/dataset/test'

# Data augmentation
train_datagen = image.ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_datagen = image.ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = image.ImageDataGenerator(
    rotation_range=15,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Image addressing
train_generator = train_datagen.flow_from_directory(
    train_set,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    val_set,
    target_size=(224, 224),
    batch_size=8,
    shuffle=True,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_set,
    target_size=(224, 224),
    batch_size=8,
    class_mode='binary'
)

# Base model
base_for_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)

for layer in base_for_model.layers:
    layer.trainable = False

# Model
model = Sequential()
model.add(base_for_model)
model.add(GaussianNoise(0.25))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

# Callbacks
mp = tf.keras.callbacks.ModelCheckpoint(filepath='mymodel.hdf5', verbose=2, save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3)
lr_schedule = LearningRateScheduler(lambda epoch, lr: lr * tf.math.exp(-0.1 * epoch))
tensorboard_callback = TensorBoard(log_dir="./logs")
callback = [es, mp, lr_schedule, tensorboard_callback]

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=80,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callback
)

# Visualization
plt.figure(figsize=(15, 10))

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot precision
plt.subplot(2, 2, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.title('Precision Comparison')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

# Plot recall
plt.subplot(2, 2, 4)
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Recall Comparison')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()

# Model evaluation
model.evaluate(train_generator)
model.evaluate(validation_generator)
model.evaluate(test_generator)

# Confusion matrix
predictions = model.predict(test_generator)
cm = confusion_matrix(test_generator.classes, predictions.round())

# Plot confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['Normal', 'SCC'], yticklabels=['Normal', 'SCC'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification report
print(classification_report(test_generator.classes, predictions.round()))


