
from fastapi import FastAPI, File, UploadFile
from model import *
import uuid
from PIL import Image 
from io import BytesIO
import ssl
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# @app.get("/predict/image")
# async def root(file: UploadFile = File(...)):
#     extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
#     return {"message": "Hello World"}
# import uvicorn
# from fastapi import FastAPI, File, UploadFile
# from application.components import predict, read_imagefile
# from model import * 
# app = FastAPI()



@app.get("/home")
async def hello():
    return "welcome hacker!" 

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...) 
):
    image = read_file_as_image(await file.read())
    image = np.asarray(image)
    
    image = image[...,0:3]

    hack = NoisyImage(image,pretrained_model,decode_predictions).hacked_image_defined_eps()

    return {
        'fake_class': str(hack[0][1]),
        'confidence': str(hack[0][2]),
        'class' : str(hack[1][1]),
        'score' : str(hack[1][2]) 
    }


@app.post("/predicts/{eps}")
async def predict(eps:float,
    file: UploadFile = File(...) 
):
    image = read_file_as_image(await file.read())
    image = np.asarray(image)
    
    image = image[...,0:3]

    hack = NoisyImage(image,pretrained_model,decode_predictions).hacked_image(eps)

    return {
        'fake_class': str(hack[0][1]),
        'confidence': str(hack[0][2]),
        'class' : str(hack[1][1]),
        'score' : str(hack[1][2]) 
    }
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


