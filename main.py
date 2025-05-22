from flask import Flask, render_template, request, flash, redirect, url_for
from flask_socketio import SocketIO, emit
import mediapipe as mp
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from gtts import gTTS

# Initialize Flask and SocketIO
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

socketio = SocketIO(app, cors_allowed_origins="*")

# Load pre-trained model
model = load_model("my_model.h5")
vowel= load_model("vowel.h5")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# list of char
nepali_characters = [
  "क", "ख", "ग", "घ", "ङ",
    "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण",
    "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म",
    "य", "र", "ल", "व",
    "श", "ष", "स", "ह",
    "क्ष", "त्र", "ज्ञ","अ", "आ", "इ",
     "ई", "उ", "ऊ", "ओ", "औ", "अं", "अः","d","c","ch"
]
vowel_char= ['j',"ा", "ि", "ी", "ु", "ै", "ो", "ौ", "ं", "ः"," "]
# Mediapipe 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)



def allowed_file(filename):
    """Check if the file is an allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_landmarks(landmarks, canvas_size=400, scale_factor=0.6):
    """Normalize hand landmarks for consistent input size."""
    x_min = min([lm[0] for lm in landmarks])
    y_min = min([lm[1] for lm in landmarks])
    x_max = max([lm[0] for lm in landmarks])
    y_max = max([lm[1] for lm in landmarks])

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    hand_width = x_max - x_min
    hand_height = y_max - y_min
    scale = scale_factor * min(canvas_size / hand_width, canvas_size / hand_height)
    offset_x = canvas_size // 2 - int(center_x * scale)
    offset_y = canvas_size // 2 - int(center_y * scale)

    normalized_landmarks = [
        (int(lm[0] * scale + offset_x), int(lm[1] * scale + offset_y)) for lm in landmarks
    ]
    return normalized_landmarks


def process_uploaded_image(filepath):
    """Process uploaded image for prediction."""
    if isinstance(filepath, str):
        image = cv2.imread(filepath)
    elif isinstance(filepath, np.ndarray):
        image = filepath
    else:
        raise ValueError("Unsupported image format.")

    if image is None:
        return None, None

    # Flip and convert to RGB
    image = cv2.flip(image, 1)
    
    # Convert to grayscale (model expects single channel image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray_image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # For hand detection

    canvas = np.ones((400, 400), np.uint8) * 255  # White canvas for hand landmarks
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        normalized_landmarks = normalize_landmarks(landmarks, canvas_size=400)

        # Drawing the landmarks and connections
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
        ]
        for start, end in connections:
            cv2.line(canvas, normalized_landmarks[start], normalized_landmarks[end], (0, 0, 0), 2)
        for lm in normalized_landmarks:
            cv2.circle(canvas, lm, 3, (0, 0, 255), -1)

        # Resize the canvas to 128x128 and add a channel dimension for grayscale
        input_image = cv2.resize(canvas, (128, 128)).reshape(1, 128, 128, 1) / 255.0
        return input_image, canvas
    return None, None


# Routes ko parto
@app.route('/', endpoint='main')
def home():
    return render_template('index.html')


@app.route('/symbols', endpoint='symbols')
def symbols():
    symbols_path = os.path.join('static', 'symbols')
    
    if not os.path.exists(symbols_path):
        return render_template('symbols.html', symbols=[])

    # Load filenames and sort them based on `nepali_characters`
    symbols_files = os.listdir(symbols_path)
    sorted_symbols = []

    for i, char in enumerate(nepali_characters):
        filename = f"{i}.jpg" 
        if filename in symbols_files:
            sorted_symbols.append((filename, char))  # Store filename with corresponding Nepali character

    return render_template('symbols.html', symbols=sorted_symbols)

@app.route('/live_detection', endpoint='live_detection')
def live_detection():
    return render_template('live_detection.html')


@socketio.on('video_frame')
def handle_video_frame(data):
    """Handle video frame from client."""
    try:
        # Convert buffer to image
        np_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        input_image, canvas = process_uploaded_image(frame)
        if input_image is not None:
            prediction = model.predict(input_image)[0]
            predicted_class = np.argmax(prediction)
            predicted_character = nepali_characters[predicted_class]

            emit('prediction', {'character': predicted_character})
        else:
            emit('prediction', {'character': "No Prediction"})
    except Exception as e:
        emit('error', {'message': str(e)})


@app.route('/upload', methods=['GET', 'POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Handle image upload for batch prediction."""
    filenames, predictions = [], []
    vowel_added = False  
    audio_filename = None  # Ensure audio_filename is always defined

    if request.method == 'POST':
        files = request.files.getlist('files')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                filenames.append(filename)

                input_image, _ = process_uploaded_image(filepath)
                if input_image is not None:
                    # Predict the character using the main model
                    prediction = model.predict(input_image)[0]
                    predicted_class = np.argmax(prediction)
                    predicted_character = nepali_characters[predicted_class]

                    if predicted_character == "ch":
                        vowel_added = True
                    else:
                        if vowel_added:
                            vowel_prediction = vowel.predict(input_image)[0]
                            vowel_class = np.argmax(vowel_prediction)
                            predicted_vowel = vowel_char[vowel_class]
                            predictions.append(predicted_vowel)
                            vowel_added = False  
                        else:
                            predictions.append(predicted_character)

                else:
                    predictions.append("")

        # Merge final word
        merged_word = "".join(predictions)
        try:
            if merged_word:
                # Generate audio only if there's a valid word
                tts = gTTS(merged_word, lang='ne')
                audio_filename = "merged_prediction.mp3"
                audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                tts.save(audio_filepath)
        except Exception as e:
            flash(f"Error generating audio: {e}")
            audio_filename = None

        return render_template('upload.html', filenames=filenames, merged_word=merged_word, audio_filename=audio_filename)

    return render_template('upload.html', filenames=[], merged_word="", audio_filename=None)


@app.route('/live_word_detection.html', endpoint='live_word_detection')
def live_word_detection():
    return render_template('live_word_detection.html')
vowel_added = False
last_character = ""
final_word=[] # Store the last predicted character for combining with vowels

@socketio.on('capture_image')
def handle_capture_image(image_data):
    global vowel_added, last_character,final_word

    try:
        # Decode image from client
        np_arr = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Process frame and make predictions
        input_image, _ = process_uploaded_image(frame)
        if input_image is not None:
            if vowel_added:
                # Predict the vowel using the secondary model
                vowel_prediction = vowel.predict(input_image)[0]
                vowel_class = np.argmax(vowel_prediction)
                predicted_vowel = vowel_char[vowel_class]
                
                # Combine the last character with the vowel
                full_character = last_character + predicted_vowel
                final_word[-1]=full_character
                vowel_added = False  # Reset the flag
                last_character = ""  # Reset the last character
                
                emit('image_prediction', {'character': predicted_vowel})
            else:
                # Predict the primary character using the main model
                prediction = model.predict(input_image)[0]
                predicted_class = np.argmax(prediction)
                predicted_character = nepali_characters[predicted_class]

                # Check if ch the predicted character requires a vowel
                if predicted_character == "ch":
                    vowel_added = True  # Set the flag for the next image
                else:
                    # Normal character prediction
                    final_word.append(predicted_character)
                    last_character = predicted_character  # Store the current character

                    emit('image_prediction', {'character': predicted_character})
        else:
            emit('image_prediction', {'character': ""})
    except Exception as e:
        emit('error', {'message': str(e)})


@socketio.on('generate_sound')
def handle_generate_sound(word):
    try:
        tts = gTTS(word, lang='ne')
        audio_filename = "word_prediction.mp3"
        audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        tts.save(audio_filepath)

        # Send audio path back 
        emit('sound_generated', {'audioPath': f"/{audio_filepath}"})
        
    except Exception as e:
        emit('error', {'message': str(e)})


if __name__ == '__main__':
    socketio.run(app, debug=True)