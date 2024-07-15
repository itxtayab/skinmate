from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from io import BytesIO
import base64

# Initialize FastAPI
app = FastAPI()

# Load your pre-trained model
base_model = tf.keras.applications.MobileNetV2(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
base_model.trainable = False
x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
model.load_weights('15july_weights.h5')

# Define class names for prediction
class_names = ['Acne and Rosacea Photos','Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
               'Nail Fungus and other Nail Disease','Psoriasis pictures Lichen Planus and related diseases',
               'Seborrheic Keratoses and other Benign Tumors']

# Define request body model
class ImageRequest(BaseModel):
    base64: str

# Define endpoint
@app.post("/predict/")
async def predict_image(image_request: ImageRequest):
    try:
        # Decode base64 string into image bytes
        image_data = base64.b64decode(image_request.base64)
        image = keras_image.load_img(BytesIO(image_data), target_size=(300, 300))
        img_array = keras_image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
        img_array /= 255.0  # Preprocess image

        # Get predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        class_confidence = predictions[0][predicted_class]

        # Return prediction result
        return {
            "Predicted Class": class_names[predicted_class],
            "Confidence": float(class_confidence)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

