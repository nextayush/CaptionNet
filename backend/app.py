import shutil
import os
import sys  # <--- Make sure sys is imported
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Import local modules
try:
    from backend import service, schemas
except ImportError:
    import service, schemas

app = FastAPI(
    title="CaptionNet API",
    version="1.0.0",
    description="Backend for Image Captioning Project"
)

# --- CORS CONFIGURATION ---
# Allows React (localhost:5173) to talk to FastAPI (localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- EVENTS ---
@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts."""
    try:
        service.load_ai_model()
    except Exception as e:
        print(f"Failed to start AI Service: {e}")
        # We don't exit here so you can still hit the root endpoint for debugging

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "online", "service": "Image Captioning Backend"}

@app.post("/predict", response_model=schemas.CaptionResponse)
async def predict(file: UploadFile = File(...), strategy: str = "beam"):
    
    # 1. Validation
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPG or PNG.")

    # 2. Save Temp File
    # We must save to disk because VGG16 expects a file path
    temp_filename = f"temp_{file.filename}"
    temp_path = BASE_DIR / "backend" / temp_filename
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 3. Call Service Layer
        caption_text = service.generate_caption(str(temp_path), strategy)
        
        return {
            "filename": file.filename,
            "caption": caption_text,
            "strategy": strategy
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Cleanup
        if temp_path.exists():
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)