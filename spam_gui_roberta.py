import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import joblib
import torch
import ipywidgets as widgets
from IPython.display import display
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def clean_text(text):
    return text.lower().strip()

tokenizer = RobertaTokenizer.from_pretrained('roberta-base-phishing')
model = RobertaForSequenceClassification.from_pretrained('roberta-base-phishing')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_prediction(input_text):
    inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_label = predictions.item()
    result = "Phishing" if predicted_label == 1 else "Safe"
    return result

def record_and_predict():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("🎙️ Listening... You have 30 seconds to talk.")
        audio = r.listen(source, timeout=60, phrase_time_limit=180)

    try:
        text = r.recognize_google(audio)
        transcription_box.delete("1.0", tk.END)
        transcription_box.insert(tk.END, text)

        prediction = get_prediction(text)
        print(prediction)
        if prediction == "Phishing":
            result_label.config(text="❌ Spam Detected!", fg="red")
        else:
            result_label.config(text="✅ Not Spam", fg="green")

    except sr.UnknownValueError:
        messagebox.showerror("Error", "Couldn't understand audio.")
    except sr.RequestError:
        messagebox.showerror("Error", "Speech recognition service unavailable.")

# GUI Layout
root = tk.Tk()
root.title("TrustTalk: Scam Call Detector")
root.geometry("500x300")

tk.Label(root, text="Click to Speak", font=("Helvetica", 16)).pack(pady=10)

record_btn = tk.Button(root, text="🎤 Record", font=("Helvetica", 14), command=record_and_predict)
record_btn.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 10))
status_label.pack()

tk.Label(root, text="Transcribed Text:", font=("Helvetica", 12)).pack()
transcription_box = tk.Text(root, height=4, width=50)
transcription_box.pack()

result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

root.mainloop()

