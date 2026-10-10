import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_learning_model(base_model, num_classes):
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_mobilenet(num_classes, img_size=(224, 224)):
    base = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    return build_transfer_learning_model(base, num_classes)

def build_vgg16(num_classes, img_size=(224, 224)):
    base = tf.keras.applications.VGG16(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    return build_transfer_learning_model(base, num_classes)

def build_resnet50(num_classes, img_size=(224, 224)):
    base = tf.keras.applications.ResNet50(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    return build_transfer_learning_model(base, num_classes)