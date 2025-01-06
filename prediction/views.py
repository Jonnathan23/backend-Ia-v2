import os
import io
import numpy as np
import requests
import gdown
from PIL import Image
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# URL pública del archivo en Google Drive
# Ruta para guardar el modelo descargado
MODEL_PATH = os.path.join("media", "modelo_multilabel4.keras")

# ID del archivo en Google Drive
FILE_ID = "1EqkvESvK26GS9xV_ULDxrSUlivWJdRMt"

# URL de descarga directa desde Google Drive
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Función para descargar el modelo si no existe
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Descargando modelo desde Google Drive...")
        gdown.download(URL, MODEL_PATH, quiet=False)
        print("Descarga completada.")
    else:
        print("El modelo ya existe en la ruta especificada.")

# Llama a la función para asegurarte de que el modelo esté disponible
download_model()

# Carga el modelo
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")

# Etiquetas de clase
LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesar la imagen para hacer predicciones.
    - Redimensiona la imagen al tamaño objetivo.
    - Normaliza los valores entre 0 y 1.
    - Añade una dimensión para batch.
    """
    image = image.resize(target_size)
    img_array = img_to_array(image) / 255.0  # Redimensionar y normalizar
    return np.expand_dims(img_array, axis=0)  # Añadir dimensión batch

def predict_image(image):
    """
    Realizar predicción sobre una imagen.
    - Convierte la imagen en una entrada válida para el modelo.
    - Retorna las clases predichas como etiquetas.
    """
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)  # Predicción del modelo
    prediction_binary = (prediction > 0.5).astype(int)  # Convertir a etiquetas binarias
    return [LABELS[i] for i, pred in enumerate(prediction_binary[0]) if pred == 1]

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request, *args, **kwargs):
        """
        Maneja solicitudes POST para realizar predicciones.
        - Recibe un archivo de imagen en la solicitud.
        - Retorna las etiquetas predichas como respuesta JSON.
        """
        try:
            # Verificar que se envió una imagen
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({"error": "No se proporcionó un archivo de imagen."}, status=400)

            # Abrir la imagen usando PIL
            image = Image.open(io.BytesIO(image_file.read()))

            # Realizar la predicción
            predictions = predict_image(image)
            return JsonResponse({"predictions": predictions}, status=200)

        except Exception as e:
            return JsonResponse({"error": f"Ocurrió un error: {str(e)}"}, status=500)