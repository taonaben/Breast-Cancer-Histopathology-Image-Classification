import tensorflow as tf
import tensorflow_hub as hub
import types

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
                layer.function.__globals__['base_model'] = base_model_instance
                print(f"Injected base_model into {layer.name}")
        
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_image(img_path, IMAGE_SIZE, model):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    img = img / 255.0
    img = tf.expand_dims(img, 0)

    prediction = model.predict(img, verbose=0)
    return prediction[0][0]


if __name__ == "__main__":
    IMAGE_SIZE = 224
    model = load_model_with_injection("notebooks/models/best_model.keras")
    
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    img_path = "data/raw/archive/BreaKHis_v1/histology_slides/breast/benign/SOB/fibroadenoma/SOB_B_F_14-21998CD/100X/SOB_B_F-14-21998CD-100-002.png"
    
    prediction = predict_image(img_path, IMAGE_SIZE, model)
    print(f"Prediction value: {prediction}")

    if prediction > 0.5:
        print("Predicted class is: Benign")
    else:
        print("Predicted class is: Malignant")