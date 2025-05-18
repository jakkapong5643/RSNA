from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
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
from pydantic import BaseModel

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

inputWeight = pd.read_csv('.\Data\OP\inputWeightInference.csv', header=None).values
outputWeight = pd.read_csv('.\Data\OP\outputWeightInference.csv', header=None).values
bias = pd.read_csv('.\Data\OP\bias.csv', header=None).values

def maxtwoind_mammo(x):
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

def maxtwoindclass_mammo(x):
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

class InputData(BaseModel):
    input_values: list[float]

@app.post("/predict/")
async def predict(data: InputData):
    try:
        X_new = np.array([data.input_values])

        H_new = 1 / (1 + np.exp(-(X_new @ inputWeight + np.tile(bias, (X_new.shape[0], 1)))))
        outputNew = np.dot(H_new, outputWeight)

        prediction = np.argmax(outputNew)
        confidence = np.max(outputNew)

        if prediction == 1:
            result = "No Osteoporosis 💀"
        elif prediction == 2:
            result = "Yes Osteoporosis 🦴"
        else:
            result = "Error in Prediction"
        
        return {"prediction": result, "confidence": confidence * 100}
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return {"prediction": "Error in Prediction", "confidence": 0.0}
    
@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("NEW.html", {"request": request})

def image_to_base64(image_path: str):
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    return rotated

def To_Class(x):
    return ["B", "A1", "A2", "A3", "C", "D", "A4"][x]


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
            # scores = results[0].boxes.conf.cpu().numpy()
            # class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            # names = results[0].names

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                # class_name = names[class_ids[i]] if names and class_ids[i] < len(names) else str(class_ids[i])
                # label = f"{class_name}: {scores[i]*100:.2f}%"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                # cv2.putText(image, (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            x1, y1, x2, y2 = map(int, boxes[0])
            cropped = image[y1:y2, x1:x2]
            cropped = rotate_image(cropped, 15)
            cropped_pil = Image.fromarray(cropped)
            
            transform = transforms.Compose([
                transforms.Resize((240, 240)),
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

        input_weight = pd.read_csv('.\Data\input_weight.csv', header=None).values
        output_weight = pd.read_csv('.\Data\output_weight.csv', header=None).values

        features_array = features_df.values
        H_infer = expit(np.dot(features_array, input_weight) + 4)
        output = np.dot(H_infer, output_weight)

        pred_class = np.argmax(output, axis=1)[0]
        confidence = np.max(output[0]) / np.sum(output[0])

        pil_img = Image.fromarray(image)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG")
        seg_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        return templates.TemplateResponse("NEW.html", {
            "request": request,
            "prediction": {
                "class": To_Class(pred_class),
                "confidence": round(np.max(output)/(((np.sum(output)-np.max(output))/6)+np.max(output)) * 100, 2)
            },
            "image_path": f"data:image/jpeg;base64,{seg_image_base64}"
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
