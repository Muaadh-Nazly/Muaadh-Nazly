from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import os

app = FastAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)  # Create a folder to store datasets
DATA_FILE = f"{DATA_DIR}/dataset.csv"

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.post("/upload_dataset/")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset (CSV file)."""
    try:
        # Save the uploaded file
        file_path = DATA_FILE
        with open(file_path, "wb") as f:
            f.write(await file.read())
        return {"message": "Dataset uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading dataset: {e}")

@app.get("/get_dataset/")
def get_dataset(format: str = "json"):
    """Retrieve the dataset in JSON or CSV format."""
    if not os.path.exists(DATA_FILE):
        raise HTTPException(status_code=404, detail="No dataset found")

    df = pd.read_csv(DATA_FILE)
    if format == "json":
        return JSONResponse(content=df.to_dict(orient="records"))
    elif format == "csv":
        return FileResponse(DATA_FILE, media_type="text/csv", filename="dataset.csv")
    else:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'csv'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
