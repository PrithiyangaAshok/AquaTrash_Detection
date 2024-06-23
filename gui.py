import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('trash_detection_model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to classify the image
def classify_image():
    global img_label
    image_path = filedialog.askopenfilename()
    if image_path:
        preprocessed_image = preprocess_image(image_path)
        prediction = model.predict(preprocessed_image)
        if prediction < 0.5:
            result = "Not Trash"
        else:
            result = "Trash"

        img = Image.open(image_path)
        img = img.resize((250, 250), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img

        messagebox.showinfo("Result", result)

# Set up the GUI
root = tk.Tk()
root.title("Trash Detection")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

btn = tk.Button(frame, text="Select Image", command=classify_image)
btn.pack(pady=10)

img_label = tk.Label(frame)
img_label.pack()

root.mainloop()
