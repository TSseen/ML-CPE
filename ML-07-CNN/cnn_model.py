from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom

def build_model(input_shape=(48, 48, 3)):
    inputs = Input(shape=input_shape)
    
    # --- 1. Data Augmentation: เพิ่มความหลากหลายให้รูปภาพ ---
    # สุ่มพลิกซ้ายขวา, เอียงรูปเล็กน้อย, และซูมเข้าออก เพื่อให้โมเดลชินกับทุกมุมกล้อง
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)
    x = RandomZoom(0.1)(x)

    # --- 2. Feature Extraction: สกัดลักษณะเด่น (เพิ่มความลึกให้โมเดล) ---
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    # เพิ่มชั้น 256 เข้ามาเพื่อให้โมเดลจับรายละเอียดที่ซับซ้อนขึ้นอย่างริ้วรอยได้
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x) 
    x = MaxPooling2D((2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    # กิ่งที่ 1: ทายเพศ (Classification) - เพิ่มเลเยอร์ให้หนาขึ้น
    branch_gender = Dense(128, activation='relu')(x)
    branch_gender = Dropout(0.4)(branch_gender)
    branch_gender = Dense(64, activation='relu')(branch_gender) # เพิ่มชั้นคิดวิเคราะห์อีกชั้น
    branch_gender = Dropout(0.3)(branch_gender)
    gender_out = Dense(1, activation='sigmoid', name='gender_out')(branch_gender)
    
    # กิ่งที่ 2: ทายอายุ (เพิ่ม Dropout กันการท่องจำ)
    branch_age = Dense(64, activation='relu')(x)
    branch_age = Dropout(0.3)(branch_age)
    age_out = Dense(1, activation='linear', name='age_out')(branch_age)

    model = Model(inputs=inputs, outputs=[gender_out, age_out])
    return model