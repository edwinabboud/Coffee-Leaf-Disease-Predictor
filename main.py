from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
app=FastAPI()

MODEL=tf.keras.models.load_model("/Users/edwinblanco/Desktop/ML_projects/Coffee-Disease-Prediction/saved_models/model.h5")
CLASS_NAMES=["Healthy","Miner","Rust"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image=read_file_as_image(await file.read())
    image=cv2.resize(image,dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    img_batch=np.expand_dims(image, 0)
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return {'class':predicted_class,
            'confidence': float(confidence)
            }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)