import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import pandas as pd
import random

# 1. Create result folder
def create_new_result_folder(base_folder='./hasil_latih'):
    os.makedirs(base_folder, exist_ok=True)
    folder_number = 1
    while os.path.exists(os.path.join(base_folder, f'hasil_latihan_{folder_number}')):
        folder_number += 1
    new_folder = os.path.join(base_folder, f'hasil_latihan_{folder_number}')
    os.makedirs(new_folder)
    print(f"Folder created: {new_folder}")
    return new_folder

# 2. Display sample images
def display_sample_images(generator, class_names, save_folder):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle('Sample Images from Each Category', fontsize=16)

    num_classes_to_display = min(len(class_names), 12)  # Limit to 12 classes

    for i, class_name in enumerate(class_names[:num_classes_to_display]):
        img_folder = os.path.join(generator.directory, class_name)
        img_files = os.listdir(img_folder)
        
        if not img_files:
            print(f"No images found in {img_folder}")
            continue
        
        img_file = random.choice(img_files)  # Random sample
        img_path = os.path.join(img_folder, img_file)

        # Load the image with error handling
        try:
            print(f"Loading image from: {img_path}")
            img = image.load_img(img_path, target_size=(img_size, img_size))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image

            ax = axes[i // 4, i % 4]
            ax.imshow(img_array)
            ax.axis('off')
            ax.set_title(class_name)

        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'sample_images.png'))
    plt.show()  # Show the figure to display it


# # 3. Define and compile the AlexNet model
# Define the AlexNet model


def build_alexnet_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (7, 7), strides=2, activation='relu', input_shape=input_shape, kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D((3, 3), strides=2),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling2D(),  # Replaces Flatten() and Dense(4096)
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    # Ringkasan arsitektur model
    model.summary()
    return model


# 4. Train the model
def train_model(model, train_generator, validation_generator, save_folder, batch_size, epochs):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping],
        steps_per_epoch=steps_per_epoch,  # Sesuaikan dengan jumlah sampel
        validation_steps=validation_steps  # Sesuaikan dengan jumlah sampel validasi
    )

    model.save(os.path.join(save_folder, 'pest_classification_alexnet_optimized.h5'))
    return history

# 5. Evaluate the model
def evaluate_model(model, validation_generator):
    val_loss, val_accuracy = model.evaluate(validation_generator)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    return val_loss, val_accuracy

# 6. Plot confusion matrix
def plot_confusion_matrix(validation_generator, y_pred, save_folder):
    cm = confusion_matrix(validation_generator.classes, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(),
                yticklabels=validation_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))
    plt.show()

# 7. Display classification report as a table and save it
def plot_classification_report_as_table(validation_generator, y_pred, target_names, save_folder):
    # Membuat classification report dalam bentuk dictionary
    report = classification_report(validation_generator.classes, y_pred, target_names=target_names, output_dict=True)
    
    # Mengonversi classification report menjadi DataFrame
    df_classification_report = pd.DataFrame(report).transpose()

    # Membuat figure dan axes untuk tabel
    fig, ax = plt.subplots(figsize=(12, 6))  # Sesuaikan ukuran figure sesuai kebutuhan

    # Menyembunyikan axes
    ax.axis('tight')
    ax.axis('off')

    # Membuat tabel menggunakan DataFrame
    table = ax.table(cellText=df_classification_report.values,
                     colLabels=df_classification_report.columns,
                     rowLabels=df_classification_report.index,
                     cellLoc='center', loc='center')

    # Menyesuaikan ukuran font agar lebih sesuai
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Sesuaikan skala tabel agar tidak terpotong

    # Adjust the layout using subplots_adjust (mirip dengan slider)
    plt.subplots_adjust(left=0.34, bottom=0.048, right=0.84, top=0.88, wspace=0.21, hspace=0.2)

    # Menyimpan tabel sebagai gambar
    plt.savefig(os.path.join(save_folder, 'classification_report_table_adjusted.png'), bbox_inches='tight')

    # Menampilkan tabel
    plt.show()




# 8. Plot accuracy and loss
def plot_accuracy_and_loss(history, save_folder):
    plt.figure(figsize=(12, 4))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(t * 100)}%" for t in np.arange(0, 1.1, 0.1)])
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Add labels for final accuracies
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    plt.text(len(history.history['accuracy']) - 1, train_acc, f'{train_acc * 100:.2f}%', color='blue', ha='right')
    plt.text(len(history.history['val_accuracy']) - 1, val_acc, f'{val_acc * 100:.2f}%', color='orange', ha='right')

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Add labels for final losses
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    plt.text(len(history.history['loss']) - 1, train_loss, f'{train_loss:.4f}', color='blue', ha='right')
    plt.text(len(history.history['val_loss']) - 1, val_loss, f'{val_loss:.4f}', color='orange', ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'accuracy_loss_plot.png'))
    plt.show()


# Main execution
if __name__ == "__main__":

    img_size = 224
    num_classes = 12
    batch_size = 64
    epochs = 1
    save_folder = create_new_result_folder()

    # Data Augmentation
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,  # Mengurangi rotasi agar tetap sesuai dengan fitur penting gambar
        width_shift_range=0.1,  # Mengurangi pergeseran agar tidak terlalu ekstrem
        height_shift_range=0.1,
        shear_range=0.15,  # Membuat shear lebih rendah untuk menghindari distorsi berlebihan
        zoom_range=0.15,  # Zoom yang sedikit lebih kecil
        horizontal_flip=True,  # Masih tetap efektif untuk augmentasi
        brightness_range=[0.8, 1.2],  # Menambah variasi pencahayaan pada gambar
        fill_mode='nearest',
        validation_split=0.2  # 20% gambar digunakan untuk validasi
    )

    train_generator = train_datagen.flow_from_directory(
        './Dataset/pest_padi',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        './Dataset/pest_padi',
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    display_sample_images(train_generator, list(train_generator.class_indices.keys()), save_folder)

    model = build_alexnet_model((img_size, img_size, 3), num_classes)
    history = train_model(model, train_generator, validation_generator, save_folder, batch_size, epochs)

    val_loss, val_accuracy = evaluate_model(model, validation_generator)

    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    plot_confusion_matrix(validation_generator, y_pred, save_folder)
    # Mendapatkan target_names
    target_names = list(validation_generator.class_indices.keys())

    # Memanggil fungsi untuk membuat tabel classification report
    plot_classification_report_as_table(validation_generator, y_pred, target_names, save_folder)
    plot_accuracy_and_loss(history, save_folder)

    print(f"All results saved in folder: {save_folder}")




# ////////////////////////////////////////////////////////////////////////



# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns
# import os
# import random

# # Function to create a new directory for results with sequential numbering
# def create_new_result_folder(base_folder='./hasil_latih'):
#     # Create base folder if it doesn't exist
#     os.makedirs(base_folder, exist_ok=True)

#     # Check existing folders and find the next available number
#     folder_number = 1
#     while os.path.exists(os.path.join(base_folder, f'hasil_latihan_{folder_number}')):
#         folder_number += 1
    
#     # Create the new folder
#     new_folder = os.path.join(base_folder, f'hasil_latihan_{folder_number}')
#     os.makedirs(new_folder)
    
#     print(f"Folder created: {new_folder}")
#     return new_folder

# # Image size and model configuration
# img_size = 224
# num_classes = 12
# batch_size = 64
# epochs = 3

# # Create a new result folder
# save_folder = create_new_result_folder()

# # Data Augmentation and Image Preprocessing
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest',
#     validation_split=0.2  # Split data into 80% train, 20% validation
# )

# # Load training and validation data
# train_generator = train_datagen.flow_from_directory(
#     './Dataset/pest_padi',  # Replace with actual dataset path
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='training'
# )

# validation_generator = train_datagen.flow_from_directory(
#     './Dataset/pest_padi',  # Replace with actual dataset path
#     target_size=(img_size, img_size),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'
# )

# # Display sample images from each category
# # Display sample images from each category and save the image
# def display_sample_images(generator, class_names, save_folder):
#     fig, axes = plt.subplots(3, 4, figsize=(12, 8))
#     fig.suptitle('Sample Images from Each Category', fontsize=16)
    
#     for i, class_name in enumerate(class_names):
#         img_folder = os.path.join(generator.directory, class_name)
#         img_file = random.choice(os.listdir(img_folder))  # Random sample
#         img_path = os.path.join(img_folder, img_file)
        
#         img = plt.imread(img_path)
#         ax = axes[i // 4, i % 4]
#         ax.imshow(img)
#         ax.axis('off')
#         ax.set_title(class_name)
    
#     plt.tight_layout()

#     # Save the plot as a PNG file before showing it
#     plt.savefig(os.path.join(save_folder, 'sample_images.png'))

#     # Show the plot
#     plt.show()

# # Display and save sample images
# display_sample_images(train_generator, list(train_generator.class_indices.keys()), save_folder)

# # AlexNet Model Architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(img_size, img_size, 3), kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
#     tf.keras.layers.MaxPooling2D((3, 3), strides=2),
#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
#     tf.keras.layers.MaxPooling2D((3, 3), strides=2),
#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
#     tf.keras.layers.MaxPooling2D((3, 3), strides=2),
#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(4096, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(4096, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])

# # Compile the model
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# # Ringkasan arsitektur model
# model.summary()

# # Early stopping to avoid overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Train the model
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     validation_data=validation_generator,
#     callbacks=[early_stopping],
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_steps=validation_generator.samples // batch_size
# )

# # Evaluate the model on validation data
# val_loss, val_accuracy = model.evaluate(validation_generator)
# print(f"Validation Loss: {val_loss:.4f}")
# print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# # Predict on validation data
# Y_pred = model.predict(validation_generator)
# y_pred = np.argmax(Y_pred, axis=1)

# # Confusion Matrix
# cm = confusion_matrix(validation_generator.classes, y_pred)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=validation_generator.class_indices.keys(),
#             yticklabels=validation_generator.class_indices.keys())
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')

# # Optimize layout
# plt.tight_layout()

# # Save the confusion matrix
# plt.savefig(os.path.join(save_folder, 'confusion_matrix.png'))

# # Show the plot
# plt.show()


# # Classification Report
# print("Classification Report:")
# target_names = list(validation_generator.class_indices.keys())
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# # Plot training & validation accuracy and loss values
# plt.figure(figsize=(12, 4))

# # Plot accuracy
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy (%)')
# plt.xlabel('Epoch')
# plt.ylim(0, 1)  # Set limits from 0 to 1 for proportion
# plt.yticks(np.arange(0, 1.1, 0.1), [f"{int(t * 100)}%" for t in np.arange(0, 1.1, 0.1)])  # Convert to percentage
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Save the plot as a PNG file
# plt.savefig(os.path.join(save_folder, 'accuracy_plot.png'))

# # Save the model
# model.save(os.path.join(save_folder, 'pest_classification_alexnet.h5'))

# print(f"All results saved in folder: {save_folder}")
