from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io
import time

app = FastAPI(title="Breast Cancer Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model
def load_model_with_injection(model_path):
    print("Loading model from:", model_path)

    # Create the base model
    base_model_instance = hub.KerasLayer(
        "https://tfhub.dev/google/efficientnet/b0/feature-vector/1",
        trainable=False,
        name="efficientnetv2-b0",
    )

    try:
        # Load the model
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False,
        )

        # Find the Lambda layer and inject base_model into its function's globals
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Lambda):
                # Inject base_model into the lambda function's global namespace
                layer.function.__globals__["base_model"] = base_model_instance
                print(f"Injected base_model into {layer.name}")

        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        print("Model loading process completed.")


model = load_model_with_injection("notebooks/models/best_model.keras")

IMAGE_SIZE = 224


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        start_time = time.time()

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, 0)

        prediction = model.predict(img_array, verbose=0)[0][0]

        classification = "Benign" if prediction > 0.5 else "Malignant"
        confidence = float(prediction if prediction > 0.5 else 1 - prediction)

        processing_time = round(time.time() - start_time, 2)

        return {
            "classification": classification,
            "confidence": round(confidence * 100, 2),
            "raw_prediction": float(prediction),
            "processing_time": processing_time,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
