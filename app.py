from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response, HTMLResponse
import uvicorn
import os
import PyPDF2
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Initialize globally so the Multi-GB PyTorch model loads only ONCE on startup
try:
    prediction_obj = PredictionPipeline()
except Exception as e:
    print(f"Warning: Failed to load Prediction Pipeline proactively. Error: {e}")
    prediction_obj = None

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !! ")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/predict")
async def predict_route(file: UploadFile = File(None), text_input: str = Form(None)):
    try:
        text = ""
        # 1. Parse Document Upload or Raw Text
        if file and file.filename:
            if file.filename.endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(file.file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + " "
            else:
                content = await file.read()
                text = content.decode("utf-8")
        elif text_input:
            text = text_input
        
        if not text.strip():
            return {"summary": "Error: Please upload a file or paste text to summarize."}

        # 2. Invoke Text Summarizer Logic
        global prediction_obj
        if prediction_obj is None:
            prediction_obj = PredictionPipeline()
            
        summary = prediction_obj.predict(text)
        
        return {"summary": summary}
    except Exception as e:
        return {"summary": f"Error processing document: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)