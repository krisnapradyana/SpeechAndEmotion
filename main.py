from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import json
import threading
from transformers import pipeline
import uvicorn
import gradio_ui  # Assuming gradio.py is named gradio_ui.py

app = FastAPI()

# Load emotion detection model
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

# In-memory chat history (Resets when server restarts)
chat_history = []

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    emotions: dict

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Construct full conversation history
        formatted_history = "\n".join(chat_history[-5:])  # Keep the last 5 exchanges
        full_prompt = f"{formatted_history}\nUser: {request.prompt}\nAI:"

        # Run Ollama via command line with context
        ollama_command = ["ollama", "run", "llama3.1", full_prompt]
        process = subprocess.run(ollama_command, capture_output=True, text=True)
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail="Ollama command failed")
        
        response_text = process.stdout.strip()
        
        # Analyze emotions in response
        emotion_scores = emotion_model(response_text)
        
        # Convert emotion output to dictionary
        emotions = {entry['label']: entry['score'] for entry in emotion_scores[0]}

        # Store response in history
        chat_history.append(f"User: {request.prompt}")
        chat_history.append(f"AI: {response_text}")
        
        return ChatResponse(response=response_text, emotions=emotions)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_gradio():
    gradio_ui.demo.launch()

if __name__ == "__main__":
    threading.Thread(target=start_gradio, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
