from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from fastapi.responses import RedirectResponse
import base64
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import torch
from torchvision import transforms, models
from PIL import Image
from scipy.special import expit
import cv2
from io import BytesIO

app = FastAPI()

vgg19 = models.vgg19(pretrained=True)
vgg19.eval()

app.mount("/fonts", StaticFiles(directory="static/fonts"), name="fonts")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/index.html", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/blog-single.html", response_class=HTMLResponse)
async def research(request: Request):
    return templates.TemplateResponse("blog-single.html", {"request": request})

@app.get("/contact.html", response_class=HTMLResponse)
async def contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/404.html", response_class=HTMLResponse)
async def not_found(request: Request):
    return templates.TemplateResponse("404.html", {"request": request})

@app.get("/NEW.html", response_class=HTMLResponse)
async def NEW(request: Request):
    return templates.TemplateResponse("NEW.html", {"request": request})

def maxtwoind_mammo_OP(x):
    y = []
    for i in range(x.shape[0]):
        n = np.argmax(x[i, :])
        if n == 0:
            y.append([1, 0])
        elif n == 1:
            y.append([0, 1])
        else:
            print("error")
            break
    return np.array(y)

def maxtwoindclass_mammo_OP(x):
    y = []
    for i in range(x.shape[0]):
        n = x[i, :]
        if np.array_equal(n, [1, 0]):
            y.append(1)
        elif np.array_equal(n, [0, 1]):
            y.append(2)
        else:
            print("Error: Invalid prediction")
            y.append(0)
    return np.array(y)

def To_Class(x):
    return ["A1", "A2", "A3", "A4", "B", "C", "D"][x]

inputWeight_OP = pd.read_csv('./Data/OP/inputWeightInference.csv', header=None).values
outputWeight_OP = pd.read_csv('./Data/OP/outputWeightInference.csv', header=None).values
bias_OP = pd.read_csv('./Data/OP/biasInference.csv', header=None).values


@app.post("/predict_OP/")
async def predict_OP(data: dict):
    try:
        input_values_OP = data["input_values_OP"]
        X_new_OP = np.array([input_values_OP])
        
        H_new_OP = 1 / (1 + np.exp(-(X_new_OP @ inputWeight_OP + np.tile(bias_OP, (X_new_OP.shape[0], 1)))))
        outputNew_OP = np.dot(H_new_OP, outputWeight_OP)

        yNew_OP = maxtwoind_mammo_OP(outputNew_OP)
        predictionsNew_OP = maxtwoindclass_mammo_OP(yNew_OP)
        confidence_OP = np.max(outputNew_OP) * 100  # แปลงเป็นเปอร์เซ็นต์

        prediction = predictionsNew_OP[0] if predictionsNew_OP.size > 0 else 0

        result_OP = {
            1: "No Osteoporosis 💀",
            2: "Yes Osteoporosis 🦴"
        }.get(prediction, "Error in Prediction")

        return JSONResponse({
            "prediction": result_OP,
            "confidence": float(confidence_OP),
            "status": "success"
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("NEW.html", {"request": request})
    
def image_to_base64(image_path: str):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.post("/predict_RSNA", response_class=HTMLResponse)
async def predict_rsna_html(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()

        img = Image.open(BytesIO(contents))

        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img.save("temp_uploaded_image.jpg")  
        yolo_model = YOLO('./Model/segRSNA.pt')
        results = yolo_model("temp_uploaded_image.jpg")

        image = cv2.imread("temp_uploaded_image.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results[0].masks is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            x1, y1, x2, y2 = map(int, boxes[0])
            cropped = image[y1:y2, x1:x2]

            cropped_pil = Image.fromarray(cropped)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(cropped_pil).unsqueeze(0)

            with torch.no_grad():
                features = vgg19.features(img_tensor)
                features = features.view(features.size(0), -1).cpu().numpy().flatten()

            features_df = pd.DataFrame(features.reshape(1, -1))
        else:
            return templates.TemplateResponse("NEW.html", {
                "request": request,
                "error": "ไม่พบ segment ในภาพ",
                "image_path": f"data:image/jpeg;base64,{base64.b64encode(contents).decode()}"
            })

        input_weight = pd.read_csv('./Data/RSNA/RSNA_Augment_96/input_weight.csv', header=None).values
        output_weight = pd.read_csv('./Data/RSNA/RSNA_Augment_96/output_weight.csv', header=None).values

        features_array = features_df.values
        H_infer = expit(np.dot(features_array, input_weight) + 4)
        output = np.dot(H_infer, output_weight)

        pred_class = np.argmax(output, axis=1)[0]
        confidence = np.max(output[0]) / np.sum(output[0])

        return templates.TemplateResponse("NEW.html", {
            "request": request,
            "prediction": {
                "class": To_Class(pred_class),
                "confidence": round(confidence * 100, 2)
            },
            "image_path": f"data:image/jpeg;base64,{base64.b64encode(contents).decode()}"
        })

    except Exception as e:
        return templates.TemplateResponse("NEW.html", {
            "request": request,
            "error": str(e)
        })
    
@app.get("/js/{file_path:path}")
async def serve_js(file_path: str):
    return FileResponse(f"static/js/{file_path}")

@app.get("/css/{file_path:path}")
async def serve_css(file_path: str):
    return FileResponse(f"static/css/{file_path}")

@app.get("/img/{file_path:path}")
async def serve_img(file_path: str):
    return FileResponse(f"static/img/{file_path}")