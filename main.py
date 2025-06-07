from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image

app = FastAPI()

# Permitir acesso do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # substitua pelo domínio do seu frontend se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result
