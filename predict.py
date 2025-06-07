import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

model = load_model("model/skin_lesion_classifier.h5")
class_names = ['benign', 'malignant']

def predict_image(file) -> dict:
    img = Image.open(BytesIO(file)).convert("RGB")
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)[0]
    result = {
        "class": class_names[np.argmax(pred)],
        "confidence": float(pred[np.argmax(pred)])
    }
    return result
