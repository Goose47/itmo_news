from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import HttpUrl

from src.model import process_query
from schemas.request import PredictionRequest, PredictionResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        answer, reasoning, sources = await process_query(body.query)
        sources: List[HttpUrl] = [HttpUrl(url_string) for url_string in sources]

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        return response
    except ValueError as e:
        error_msg = str(e)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)