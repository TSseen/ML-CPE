import tensorflow as tf

def load_datasets(train_dir, val_dir, img_size=(224, 224), batch_size=32, selected_classes=None):
    print("กำลังโหลดชุดข้อมูล Training...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, seed=123, image_size=img_size, batch_size=batch_size,
        class_names=selected_classes 
    )

    print("กำลังโหลดชุดข้อมูล Validation...")
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, seed=123, image_size=img_size, batch_size=batch_size, shuffle=False,
        class_names=selected_classes 
    )
    
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names