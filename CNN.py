import tensorflow as tf
import os
import cv2
import imghdr

# directory for data
data_dir = "blackHairv2"
# array of img extension types
image_exts = ['jpeg', 'jpg', 'bmp', 'gif']

# Remove bad images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))

# Load data
data = tf.keras.utils.image_dataset_from_directory(data_dir, image_size=(256, 256), batch_size=32)

# Define class names
class_names = data.class_names

# Define sizes of each data partition
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

# Split data into training, validation, and testing
train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

# make model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')
model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

# Load and preprocess the test image
test_image_path = "test.jpg"
test_image = cv2.imread(test_image_path)
resized_image = cv2.resize(test_image, (256, 256))
input_image = resized_image / 255.0  # Normalize the image

# Expand dimensions to match the input shape of the model
input_image = tf.expand_dims(input_image, axis=0)

# Make predictions on the test image
predictions = model.predict(input_image)

# Get the class index with the highest probability
predicted_class_index = tf.argmax(predictions, axis=1)[0].numpy()

# Get the class label based on the index
class_labels = ['Wavy (2a - 2c)', 'Twists', 'Tight Curly (3b - 3c)', 'Tight Coily (4b - 4c)', 'Straight (1a - 1c)', 'Locs',
                'Curly (3a)', 'Coily', 'Braids']
predicted_class_label = class_labels[predicted_class_index]

print("Predicted Class:", predicted_class_label)
