from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import json
import os

app = FastAPI(title="Voice Converter API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "https://active-passive-converter.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConversionRequest(BaseModel):
    text: str
    direction: str  # "active_to_passive" or "passive_to_active"

class ConversionResponse(BaseModel):
    converted_text: str

# Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

async def convert_with_groq(text: str, direction: str) -> str:
    if direction == "active_to_passive":
        prompt = f"Convert this active voice sentence to passive voice. Only return the converted sentence, nothing else: {text}"
    else:
        prompt = f"Convert this passive voice sentence to active voice. Only return the converted sentence, nothing else: {text}"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 150
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            converted_text = data["choices"][0]["message"]["content"].strip()
            return converted_text
            
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
        except KeyError as e:
            raise HTTPException(status_code=500, detail=f"Unexpected API response format: {str(e)}")

@app.post("/api/convert", response_model=ConversionResponse)
async def convert_voice(request: ConversionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if request.direction not in ["active_to_passive", "passive_to_active"]:
        raise HTTPException(status_code=400, detail="Invalid direction")
    
    converted_text = await convert_with_groq(request.text, request.direction)
    
    return ConversionResponse(converted_text=converted_text)

@app.get("/")
async def root():
    return {"message": "Voice Converter API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)