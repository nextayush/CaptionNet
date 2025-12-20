import sys
import uuid
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- PATH SETUP ---
# Detect if we are running from 'backend' dir or root
current_dir = Path(os.getcwd())
if current_dir.name == "backend":
    # If running from backend/ folder, go up one level
    BASE_DIR = current_dir.parent
else:
    # If running from root, we are already there
    BASE_DIR = current_dir

# Add root to sys.path so we can import 'src'
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# --- IMPORT SERVICE ---
# Try/Except handles imports regardless of where you run the script from
try:
    from backend import service
except ImportError:
    import service

app = FastAPI()

# Enable CORS (Allows Frontend to talk to Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists
TEMP_DIR = BASE_DIR / "backend" / "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    """Load the model once when server starts"""
    print(f"📂 Project Root detected at: {BASE_DIR}")
    try:
        service.load_ai_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

@app.get("/")
def home():
    return {"status": "online", "message": "CaptionNet Backend is Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), strategy: str = "beam"):
    """
    Receives an image, saves it uniquely, runs AI, and cleans up.
    """
    # 1. Generate a unique filename to prevent Windows file locking conflicts
    unique_filename = f"{uuid.uuid4().hex}.jpg"
    temp_path = TEMP_DIR / unique_filename

    try:
        # 2. Save the uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Run Prediction
        caption = service.generate_caption(str(temp_path), strategy)
        
        return JSONResponse(content={"caption": caption})

    except Exception as e:
        print(f"❌ ERROR processing request: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 4. Cleanup: Delete the file regardless of success/failure
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except PermissionError:
                print(f"⚠️ Warning: Could not delete temp file {unique_filename} (Windows Lock). It will stay in temp folder.")
            except Exception as cleanup_err:
                print(f"⚠️ Cleanup error: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    # Run on localhost
    uvicorn.run(app, host="127.0.0.1", port=8000)
