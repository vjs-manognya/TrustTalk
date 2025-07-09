import google.generativeai as genai
genai.configure(api_key="AIzaSyBam890jDotYjjR6k7qJjQ8y_4Q_TEX-Fs")

import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import joblib

def clean_text(text):
    return text.lower().strip()

def predict_spam(text):
  # Load the model
  model = genai.GenerativeModel('gemini-1.5-flash-8b')

  # Define a single prediction prompt
  prompt = f"""
  Determine if the following call transcript is spam or not. Reply only with "fraud" or "normal".

    Examples:
    Transcript: "You are eligible for a business loan. Pay the processing fee now to continue."
    Label: fraud

    Transcript: "Hi, I wanted to ask about my current life insurance policy with LIC."
    Label: normal

    Transcript: "Dear customer, your KYC is pending. Click this link or your account will be blocked."
    Label: fraud
    Transcript: "{text}"
    Answer:
    """

  response = model.generate_content(prompt)
  answer=response.text.strip()
  if answer == "fraud":
    return 1
  else:
    return 0

# Mic capture + transcription
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

        prediction = predict_spam(text)
        print(prediction)
        if prediction == 1:
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
