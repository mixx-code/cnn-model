import tensorflow as tf
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from math import ceil  # Tambahkan ini untuk pembulatan ke atas

# Tentukan path ke direktori dataset
base_dir = 'Dataset/pest'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# ImageDataGenerator untuk preprocessing dan augmentasi data
# Augmentasi data untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,              # Rotasi gambar hingga 20 derajat
    width_shift_range=0.1,          # Pergeseran horizontal hingga 10% dari lebar gambar
    height_shift_range=0.1,         # Pergeseran vertikal hingga 10% dari tinggi gambar
    shear_range=0.15,               # Transformasi shear hingga 15%
    zoom_range=[0.8, 1.2],          # Zoom in/out lebih besar
    horizontal_flip=True,           # Membalik gambar secara horizontal
    vertical_flip=True,             # Tambahkan flip vertikal
    brightness_range=[0.5, 1.2],    # Mengubah kecerahan gambar antara 50% dan 120%
    fill_mode='nearest',            # Mengisi piksel hilang dengan nilai tetangga terdekat
    validation_split=0.1            # Menggunakan 50% data untuk validasi
)
# Data generator untuk validasi
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

batch_size = 8  # Contoh ukuran batch lebih kecil

# Generator untuk data training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = val_datagen.flow_from_directory(
    train_dir,
    target_size=(227, 227),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    test_dir,
    target_size=(227, 227),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Print jumlah data training dan data validasi
print(f"Jumlah data training: {train_generator.samples}")
print(f"Jumlah data validasi: {validation_generator.samples}")

# Definisi model AlexNet yang dioptimalkan
model = models.Sequential([
    layers.Conv2D(32, (11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)),
    BatchNormalization(),
    layers.MaxPooling2D((1, 1), strides=2),
    
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((1, 1), strides=2),
    
    layers.Conv2D(128, (2, 2), padding='same', activation='relu'),
    BatchNormalization(),
    layers.Conv2D(128, (1, 1), padding='same', activation='relu'),
    BatchNormalization(),
    layers.MaxPooling2D((1, 1), strides=2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.02)),  # Tingkatkan L2 regularization
    layers.Dropout(0.5),  # Tingkatkan dropout
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Kompilasi model
optimizer = optimizers.Adam(learning_rate=0.00001)  # Turunkan learning rate
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Ringkasan arsitektur model
model.summary()

# Callback untuk menghentikan pelatihan lebih awal jika tidak ada peningkatan
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)  # Tingkatkan patience
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

# Hitung steps_per_epoch dan validation_steps menggunakan pembulatan ke atas
steps_per_epoch = ceil(train_generator.samples / 4)  # Perkecil batch size menjadi 4
validation_steps = ceil(validation_generator.samples / 4)

# Latih model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluasi model pada data uji
test_loss, test_acc = model.evaluate(test_generator, steps=len(test_generator))
print("\nAkurasi pada data uji: {:.2f}%".format(test_acc * 100))
print("Loss pada data uji: {:.4f}".format(test_loss))

# Print val_accuracy dan val_loss dari history pelatihan
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']

print("\nVal Accuracy: {:.2f}%".format(val_accuracy[-1] * 100))
print("Val Loss: {:.4f}".format(val_loss[-1]))

# Prediksi data uji
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(true_classes, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualisasi Laporan Klasifikasi
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(12, 7))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm')
plt.title('Classification Report')
plt.show()

# Plot loss dan akurasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss selama pelatihan
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
