import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from io import StringIO

app = FastAPI()

@app.post("/calculate-score/")
async def calculate_score(file: UploadFile):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        # Read the CSV file
        contents = await file.read()
        csv_data = StringIO(contents.decode('utf-8'))
        df = pd.read_csv(csv_data)
        
        # Validate the required columns
        required_columns = {"job_id", "job_cost", "job_cred"}
        if not required_columns.issubset(df.columns):
            raise HTTPException(status_code=400, detail=f"CSV must contain columns: {required_columns}")
        
        # Calculate the weighted average
        sum_weighted = (df['job_cost'] * df['job_cred']).sum()
        sum_cost = df['job_cost'].sum()
        
        if sum_cost == 0:
            raise HTTPException(status_code=400, detail="Sum of job_cost cannot be zero.")
        
        result = round(sum_weighted / sum_cost, 2)
        return {"score": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("credcalc:app", host="127.0.0.1", port=8000, reload=True)