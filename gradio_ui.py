import gradio as gr
import requests
import json
import matplotlib.pyplot as plt

FASTAPI_URL = "http://localhost:8000/chat"

# Mapping emotions to corresponding emojis
EMOTION_EMOJIS = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
    "love": "😍"
}

chat_history = []

def process_chat(user_input):
    if user_input.lower() == "reset":
        chat_history.clear()
        return "Chat reset. Start a new topic!", None, "🔄"
    
    response = requests.post(FASTAPI_URL, json={"prompt": user_input})
    
    if response.status_code == 200:
        data = response.json()
        chat_response = data["response"]
        emotions = data["emotions"]
        
        # Sort emotions by highest score
        sorted_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))
        dominant_emotion = next(iter(sorted_emotions))
        
        # Generate bar chart
        fig, ax = plt.subplots()
        ax.bar(sorted_emotions.keys(), sorted_emotions.values(), color='skyblue')
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Score")
        ax.set_title("Emotion Analysis")
        plt.xticks(rotation=45)
        
        # Get the corresponding emoji based on emotion
        emotion_emoji = EMOTION_EMOJIS.get(dominant_emotion, "😐")
        
        return chat_response, fig, emotion_emoji
    else:
        return "Error communicating with backend", None, "❌"

with gr.Blocks() as demo:
    gr.Markdown("# AI Chat with Emoji Expression")
    
    with gr.Column():
        chat_input = gr.Textbox(label="Enter your message")
        with gr.Row():
            chat_button = gr.Button("Send")
            reset_button = gr.Button("Reset Chat")
    
    with gr.Column():
        chat_output = gr.Textbox(label="Chatbot Response")
        emotion_output = gr.Plot(label="Emotion Analysis")
        face_output = gr.Textbox(label="Character Emoji Expression")
    
    chat_button.click(process_chat, inputs=[chat_input], outputs=[chat_output, emotion_output, face_output])
    reset_button.click(lambda: ("Chat reset. Start a new topic!", None, "🔄"), inputs=[], outputs=[chat_output, emotion_output, face_output])

if __name__ == "__main__":
    demo.launch()
