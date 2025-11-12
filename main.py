import warnings
warnings.filterwarnings('ignore')

import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import pyjokes
import wikipedia
import requests
import time
import numpy as np

# --- ML IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# --------------------

recognizer = sr.Recognizer()

# -------------------- Text-to-Speech Engine -------------------- #
def speak(text: str):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty('voice', voices[1].id)  # Female voice
    engine.say(text)
    engine.runAndWait()

# -------------------- Listen for User Commands -------------------- #
def listen_command() -> str:
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=6)

        command = recognizer.recognize_google(audio).lower()
        command = command.replace('eva', '').replace('hey eva', '').strip()
        print(f"🗣 You said: {command}")
        return command

    except:
        return ""

# -------------------- Fetch Weather -------------------- #
def get_weather(city: str) -> str:
    api_key = "<YOUR_API_KEY>"  # Replace with your OpenWeatherMap API key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    url = f"{base_url}appid={api_key}&q={city}"

    try:
        data = requests.get(url, timeout=5).json()
        if data.get("cod") != "404":
            temp = data["main"]["temp"] - 273.15
            return str(int(temp))
        return "not found"
    except:
        return "not found"

# --- ML Training and Evaluation --- #
def train_and_evaluate_model():
    print("\n--- Training EVA's ML Model (SVM) ---")

    # Expanded labeled dataset
    commands = [
        # Play songs
        "play a song by queen", "play the latest hits", "play some jazz music",
        "play despacito", "play some music", "play song",
        "play the song shape of you", "can you play a song", "start playing music",

        # Get time
        "what time is it now", "tell me the time", "show me the time",
        "what is the time", "give me the current time", "what's the time now",
        "display time", "please tell the time", "check the time",

        # Tell joke
        "tell me a joke", "i need a good joke", "can you crack a joke",
        "make me laugh", "say something funny", "share a joke",

        # Wiki search
        "who is elon musk", "what is the internet", "tell me about quantum physics",
        "who is albert einstein", "what is artificial intelligence", "what is a black hole",

        # Get weather
        "what is the weather like in hong kong", "what is the temperature", "weather report",
        "what is the weather today", "how's the weather", "current weather conditions",

        # Exit
        "stop the program", "exit now", "goodbye eva", "quit the assistant", "terminate program"
    ]

    intents = [
        # Play songs
        *["play_song"] * 9,
        # Get time
        *["get_time"] * 9,
        # Tell joke
        *["tell_joke"] * 6,
        # Wiki search
        *["wiki_search"] * 6,
        # Get weather
        *["get_weather"] * 6,
        # Exit
        *["exit_app"] * 5
    ]

    unique_intents, counts = np.unique(intents, return_counts=True)
    print(f"Intent Distribution:\n{dict(zip(unique_intents, counts))}")

    X_train, X_test, y_train, y_test = train_test_split(
        commands, intents, test_size=0.25, random_state=42, stratify=intents
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Use SVM classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Test Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-----------------------------------------------------")

    return model, vectorizer

LR_model = None
TFIDF_vectorizer = None

# -------------------- EVA Command Handler -------------------- #
def process_eva():
    global LR_model, TFIDF_vectorizer
    command = listen_command()
    if not command:
        return True

    print("-" * 55)

    # Hybrid check: quick rule-based fix before ML prediction
    if "time" in command:
        predicted_intent = "get_time"
        confidence = 100
    elif "weather" in command or "temperature" in command:
        predicted_intent = "get_weather"
        confidence = 100
    else:
        command_vec = TFIDF_vectorizer.transform([command])
        predicted_intent = LR_model.predict(command_vec)[0]
        probs = LR_model.predict_proba(command_vec)[0]
        confidence = np.max(probs) * 100

    print(f"🧠 Predicted Intent: {predicted_intent} ({confidence:.2f}% confidence)")

    # Confidence safeguard
    if confidence < 35:
        speak("I'm not sure I understood that. Please repeat.")
        print("⚠ Low confidence prediction.")
        return True

    # -------- Execute Actions ---------- #
    if predicted_intent == 'play_song':
        song = command.replace('play', '').strip()
        speak(f"Playing {song}")
        pywhatkit.playonyt(song)

    elif predicted_intent == 'get_time':
        now = datetime.datetime.now().strftime('%I:%M %p')
        speak(f"The time is {now}")
        print(f"⏰ Time: {now}")

    elif predicted_intent == 'tell_joke':
        joke = pyjokes.get_joke()
        speak(joke)
        print(f"😂 {joke}")

    elif predicted_intent == 'wiki_search':
        query = command.replace('who is', '').replace('what is', '').strip()
        try:
            info = wikipedia.summary(query, sentences=2)
            speak(info)
            print(f"📘 {info}")
        except:
            speak("Sorry, I couldn't find information on that.")
            print("⚠ Wikipedia search failed.")

    elif predicted_intent == 'get_weather':
        city = "Hong Kong"
        temp = get_weather(city)
        if temp != "not found":
            msg = f"The temperature in {city} is {temp}°C"
            speak(msg)
            print(f"🌦 {msg}")
        else:
            speak("Unable to fetch weather details.")
            print("⚠ Weather data unavailable.")

    elif predicted_intent == 'exit_app':
        speak("Goodbye! Have a great day.")
        print("👋 EVA shutting down.")
        return False

    else:
        speak("I didn't understand that. Please repeat.")
        print("❓ Unknown Command.")

    print("-" * 55)
    return True

# -------------------- Main Program -------------------- #
if __name__ == "__main__":
    print("🤖 EVA Voice Assistant Initialized")

    LR_model, TFIDF_vectorizer = train_and_evaluate_model()

    print("Say 'stop', 'exit', or 'bye' to close the assistant.")
    print("=" * 55)

    speak("Eva started. I'm listening.")

    while True:
        try:
            if not process_eva():
                break
            time.sleep(0.3)
        except KeyboardInterrupt:
            print("\n👋 EVA stopped manually.")
            speak("Goodbye")
            break
        except Exception as e:
            print(f"⚠ General Error: {e}")
            time.sleep(1)
            continue
