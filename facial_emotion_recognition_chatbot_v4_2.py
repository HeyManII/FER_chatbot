#import threading
import time
time.clock=time.time
import cv2
import numpy as np
#import keyboard
import random
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from keras.models import load_model
import tkinter as tk
from PIL import ImageTk, Image
import cv2

from chatterbot.conversation import Statement

# Set up FER
model = load_model('model_file.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
global response

expressions = {
    "Happy": "excited",
    "Sad": "sympathetic",
    "Angry": "calm_angry",
    "Fear":"calm_fear",
    "Surprise":"calm_surprise",
    "Disgust":"calm_disgust",
    "Neutral":"neutral"
}

# Set up the chatbot
chatbot = ChatBot('Facial Expression Bot')
trainer = ListTrainer(chatbot)
# Train the chatbot on a list of responses
trainer.train([
    "Hello",
    "Hi there!",
    "How are you?",
    "I'm doing great, thanks for asking.",
    "What's your name?",
    "What can you do?",
    "I can help you recognize facial expressions and emotions",
    "How does it work?",
    "That's interesting!",
    "Thank you! Is there anything else you'd like to know?"
])


trainer.train([
    "I feel sad",
    "cheer up, Do you want to talk about it?", 
    "I don't feel good",
     "Do you know any good jokes?",
    "How about finding ABC counsellor? He can help ",
    "Yes, I lost my job",
    "I'm sorry to hear that. Have you tried looking for new opportunities?",
    "No, I don't know where to start",
    "I can help you with that. Let's work on updating your resume and searching for job postings online.",
    "Thank you, that would be great",
    "You're welcome! Don't hesitate to reach out if you need more support."
])

trainer.train([
    "I am happy",
    "that's good, can you tell me more",
    "That's great! What's making you happy?",
    "I just got promoted at work",
    "Congratulations! That's a great achievement.",
    "Thank you, I'm really excited",
    "You should be! Is there anything else you'd like to share?"
])

trainer.train([
    "I'm feeling angry",
    "I'm sorry to hear that. Can you tell me what's making you angry?"
])

trainer.train([
    "I need help",
    "What kind of help do you need?"    
])

trainer.train([
    "Goodbye",
    "Goodbye! It was nice chatting with you."    
])

trainer.train([
    "I'm feeling overwhelmed",
    "That's understandable. Can you tell me more about what's overwhelming you?"    
])

def get_response(chatbot, tone):
    # Define a list of responses for each tone
    responses = {
        "neutral": [
            "I'm sorry, I didn't quite understand that.",
            "Could you please rephrase that?",
            "I'm not sure I know what you mean.",
            "Is there something specific you'd like to talk about?",
            "How can I assist you today?",
            "Let's try to find a solution together.",
            "I'm here to help you in any way I can."
        ],
        "excited": [
            "That's great news!",
            "I'm so happy for you!",
            "That sounds amazing!",
            "That's great news!",
            "I'm so happy for you!",
            "That sounds amazing!",
            "Wow, that's fantastic!",
            "Congratulations!",
            "You must be over the moon!",
            "I'm excited to hear more about it!",
            
            #from happy
             "It's great to see you so happy!",
            "You have every reason to be happy!",
            "You deserve this happiness!",
            "I'm so glad to see you smile!",
            "Keep up the good work!",
            "What's making you so happy?",
            "I'm excited to hear more about it!",
        ],
        "sympathetic": [
            "I'm sorry to hear that.",
            "That must be tough.",
            "I'm here for you.",
            "I understand how you feel.",
            "It's okay to feel that way.",
            "You're not alone in this.",
            "Let me know how I can help.",

            #from sad
            "I'm sorry that you're feeling sad.",
            "It's okay to feel sad sometimes.",
            "Remember that you're not alone.",
            "Do you want to talk about what's making you feel sad?",
            "Take all the time you need to feel better.",
            "I'm here for you.",
            "Let me know how I can help.",
        ],
        "calm_angry": [
            "Let's take a deep breath and talk this through.",
            "I'm here to listen.",
            "I understand how you feel.",
            "That's a difficult situation.",
            "Let's work through this together.",
            "We can figure this out.",
            "It's okay to take a break and come back to this later.",

            #from angry
            "I understand that you're angry.",
            "Let's take a step back and calm down.",
            "We can work through this together.",
            "What's making you so angry?",
            "I'm sorry that you're feeling this way.",
            "Let's take a break and come back to this later.",
            "I'm here to help you find a solution.",
        ],
        
        "calm_fear": [
            "It's okay to feel afraid sometimes.",
            "Remember that you're not alone.",
            "What can we do to make you feel safer?",
            "Let's work together to overcome your fears.",
            "I'm here to help you every step of the way.",
            "Take a deep breath and focus on the present.",
            "Let's take it one step at a time.",

            #from calm
             "Let's take a deep breath and talk this through.",
            "I'm here to listen.",
            "I understand how you feel.",
            "That's a difficult situation.",
            "Let's work through this together.",
            "We can figure this out.",
            "It's okay to take a break and come back to this later.",
        ],   
        "calm_surprise": [
            "Wow, I wasn't expecting that!",
            "That's definitely a surprise!",
            "What a pleasant surprise!",
            "I'm intrigued, tell me more!",
            "I'm excited to hear what happens next!",
            "You caught me off guard!",
            "I didn't see that coming!",
        ],
        "calm_disgust": [
            "I'm sorry that you're feeling disgusted.",
            "That's a really unpleasant feeling.",
            "Let's work together to find a solution.",
            "What's making you feel this way?",
            "It's okay to feel this way, but we need to find a way to move forward.",
            "Let me know how I can help.",
            "Remember that we can work through this together."
        ], 
    }
    # Select a response randomly from the list of responses for the selected tone
    return random.choice(responses[tone])

# create tkinter window
window = tk.Tk()
window.title("GUI")

# create a frame for the left view and right view
left_frame = tk.Frame(window)
right_frame = tk.Frame(window)

# create a canvas to display the camera on the left hand side
canvas = tk.Canvas(left_frame, width=800, height=600)
canvas.pack()

# Load the facial expression detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create a video capture object for the default camera
cap = cv2.VideoCapture(0)

# define a class to store the processed Image
class VideoFeed:
    def __init__(self):
        self.photo = None
        
video_feed = VideoFeed()

# detect emotion in Image and return
def detect_expression(image):
    resized_face_image = cv2.resize(image, (48, 48))

    # Reshape the face image to match the input shape of the DeepFace model
    reshaped_face_image = np.reshape(resized_face_image, (1, 48, 48, 1))

    # Normalize the face image
    normalized_face_image = reshaped_face_image / 255.0

    # Use the model to predict the emotion of the face
    emotion_pred = model.predict(normalized_face_image)

    # Get the index of the predicted emotion label
    emotion_index = np.argmax(emotion_pred)

    # Get the predicted emotion label
    emotion_label = emotion_labels[emotion_index]

    return emotion_label

# create a frame for the chat history and input box
chat_frame = tk.Frame(right_frame)

# create a view to display the chat history
chat_history = tk.Text(chat_frame, state=tk.DISABLED, height=30, width=50)

# create a Scrollbar widget to allow scrolling
scrollbar = tk.Scrollbar(chat_frame)
scrollbar.config(command=chat_history.yview)
chat_history.config(yscrollcommand=scrollbar.set)

# create a label and text box for the user input
label = tk.Label(right_frame, text="Enter message:")
entry = tk.Entry(right_frame)

# Update the camera view
def update_camera():
    global tone
    current_expression = "neutral"
    
    # read a frame from the video capture object
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    # Loop through the detected faces
    for (x, y, w, h) in faces:            
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Extract the face region
        roi_gray = gray[y:y+h, x:x+w]

        # Detect the facial expression in the input image
        new_expression = detect_expression(roi_gray)
        # Check if the new facial expression is different from the current facial expression
        if new_expression != current_expression:
            # Update the current facial expression
            current_expression = new_expression

            if new_expression in expressions:
                tone = expressions[new_expression]
                if chat_history.index('end-1c') == '1.0' and current_expression != "neutral":
                    # send an initial message from the system
                    response = get_response(chatbot, tone)
                    chat_history.configure(state=tk.NORMAL)
                    chat_history.insert(tk.END, "System: " + response + "\n")
                    chat_history.configure
            else:
                tone = "neutral"
    
    # convert frame to RGB format and resize it
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize((800, 600))
    
    # create a new PhotoImage object from the resized frame
    frame_tk = ImageTk.PhotoImage(frame)
    
    # update the canvas with the new PhotoImage object
    canvas.create_image(0, 0, anchor=tk.NW, image=frame_tk)
    
    # add a text label to the top left corner of the camera view
    canvas.create_text(10, 10, anchor=tk.NW, text=current_expression, fill="red", font=("Arial", 20, "bold"))

    # update the VideoFeed object with the new PhotoImage object
    video_feed.photo = frame_tk
    
    # update the camera view every 0.01 second
    canvas.after(10, update_camera)

# update camera to start the camera view
update_camera()

# create a function to for messaging
# def send_message(event):

#     global tone

#     # get the message from the input box
#     message = entry.get()
    
#     # append the message to the chat history
#     chat_history.configure(state=tk.NORMAL)
#     chat_history.insert(tk.END, "You: " + message + "\n")
#     chat_history.configure(state=tk.DISABLED)
    
#     #  clear the input box
#     entry.delete(0, tk.END)
    
#     # send a response from the chatbot
#     #response = get_response(message, tone)
#     response2 = str(chatbot.generate_response(message))
#     chat_history.configure(state=tk.NORMAL)
#     chat_history.insert(tk.END, "System: " + response + "\n")
#     chat_history.configure(state=tk.DISABLED)

def send_message(event):
    message = event.widget.get()
    event.widget.delete(0, tk.END)
    statement = Statement(text=message)
    chat_history.configure(state=tk.NORMAL)
    chat_history.insert(tk.END, "You: " + message + "\n")
    chat_history.configure(state=tk.DISABLED)

    response2 = chatbot.generate_response(statement)

    chat_history.configure(state=tk.NORMAL)
    
    chat_history.insert(tk.END, "Bot: " + response2.text + "\n")
    chat_history.configure(state=tk.DISABLED)





entry.pack()
entry.bind("<Return>", send_message)     



# pack the chat history, scrollbar, and input box
chat_history.pack(side=tk.LEFT)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
entry.pack(side=tk.BOTTOM, fill=tk.X)
label.pack(side=tk.BOTTOM)

# pack the chat frame on the right side
chat_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# pack the frames on the window
left_frame.pack(side=tk.LEFT, padx=10)
right_frame.pack(side=tk.RIGHT, padx=10)

# start the GUI event loop
window.mainloop()

# release the video capture object when the window is closed
cap.release()
