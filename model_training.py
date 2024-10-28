import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

# Tentukan path ke direktori dataset
base_dir = 'Dataset/pest'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# ImageDataGenerator untuk preprocessing dan augmentasi data
# Mengubah fill_mode ke 'reflect'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='reflect',  # Ubah fill_mode ke 'reflect' atau 'wrap' untuk eksperimen
    validation_split=0.2
)


# Data generator untuk validasi
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Generator untuk data training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Generator untuk data validasi
validation_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Generator untuk data uji
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Perbarui Model CNN
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3), kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(512, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)), # Menambah layer
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Optimizer dengan learning rate lebih rendah
optimizer = optimizers.Adam(learning_rate=0.00005)

# Kompilasi model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ringkasan arsitektur model
model.summary()

# Callback untuk menghentikan pelatihan lebih awal jika tidak ada peningkatan
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Tentukan jumlah steps per epoch berdasarkan panjang data generator
steps_per_epoch = len(train_generator)  # Menggunakan jumlah batch dalam train_generator
validation_steps = len(validation_generator)  # Menggunakan jumlah batch dalam validation_generator


# Latih model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,  # Tambah epochs
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluasi model pada data uji
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("\nAkurasi pada data uji: {:.2f}%".format(test_acc * 100))

# Prediksi data uji
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Laporan Klasifikasi
print("\nLaporan Klasifikasi:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))

# Plot loss dan akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
