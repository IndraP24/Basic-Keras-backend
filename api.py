# import the necessary packages
from logging import debug
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import io
import sys


app = FastAPI(title="Keras ImageNet Web App", 
              description="A simple web application that accepts an image classifies the image made according to the ResNet50 model pre-trained for the ImageNet dataset!")
model = None



def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights='imagenet')


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


@app.get("/", tags=["root"])
async def read_root():
    return {
        "message": "Head over to /predict"
    }


@app.post("/predict", tags=["predict"])
async def predict_image(file: UploadFile = File(...)):
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    load_model()

    # Ensure that this is an image
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')

    try:
        # Read image contents
        img = await file.read()
        image = Image.open(io.BytesIO(img)).convert("RGB")

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(224, 224))

        # classify the input image and then initialize the list
        # of predictions to return to the client
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)

        data["predictions"] = []

        # loop over the results and add them to the list of
        # returned predictions
        for (imagenetID, label, prob) in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)
        
        # indicate that the request was a success
            data["success"] = True

        return {
            "success": data["success"],
            "result": data["predictions"][0]
        }

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)