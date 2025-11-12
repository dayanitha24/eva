

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

# --- ML IMPORTS ---
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# --------------------

recognizer = sr.Recognizer()

# -------------------- Text-to-Speech Engine -------------------- #
def speak(text: str):
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty('voice', voices[1].id)   # Female voice
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
    api_key = "<YOUR_API_KEY>"  
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
    print("\n---  (Logistic Regression) ---")

    # Updated labeled dataset (calculator removed)
    commands = [
        "play a song by queen", "play the latest hits", "play some jazz music",
        "what time is it now", "tell me the time", "show me the time",
        "tell me a joke", "i need a good joke", "can you crack a joke",
        "who is elon musk", "what is the internet", "tell me about quantum physics",
        "what is the weather like in hong kong", "what is the temperature", "weather report",
        "stop the program", "exit now", "goodbye eva"
    ]
    intents = [
        "play_song", "play_song", "play_song",
        "get_time", "get_time", "get_time",
        "tell_joke", "tell_joke", "tell_joke",
        "wiki_search", "wiki_search", "wiki_search",
        "get_weather", "get_weather", "get_weather",
        "exit_app", "exit_app", "exit_app"
    ]

    unique_intents, counts = np.unique(intents, return_counts=True)
    print(f"Intent Distribution:\n{dict(zip(unique_intents, counts))}")

    X_train, X_test, y_train, y_test = train_test_split(
        commands, intents, test_size=0.3, random_state=42, stratify=intents
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
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


# -------------------- EVA Command Handler 
# ----------------- #
def process_eva():
    global LR_model, TFIDF_vectorizer
    command = listen_command()
    if not command:
        return True

    print("-" * 55)

    command_vec = TFIDF_vectorizer.transform([command])
    predicted_intent = LR_model.predict(command_vec)[0]
    probs = LR_model.predict_proba(command_vec)[0]
    confidence = np.max(probs) * 100

    print(f"🧠 Predicted Intent: {predicted_intent} ({confidence:.2f}% confidence)")

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
        print(f"❓ Unknown Command.")

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
