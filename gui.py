import tkinter as tk
from tkinter import ttk, filedialog, Label
from PIL import Image, ImageTk
from functions import *
import joblib
import os

rf_classifier = joblib.load('random_forest_model.pkl')

def upload_image(file_path):
    img = preprocess_image(file_path, 240, 240)
    img = img.reshape(1, -1)
    prediction = rf_classifier.predict(img)
    showimage(file_path)
    butt_display.grid_forget()
    Label(root, text="Result: " + prediction[0], font=("Arial", 20)).grid(row=4, column=0)

def upload_wrapper():
    global file_path
    file_path = filedialog.askopenfilename()
    file_ref_path = os.path.relpath(file_path)
    for widget in root.winfo_children():
        if widget not in [text1, upload_button]:
            widget.grid_forget()
    upload_image(file_ref_path)

def showimage(file_path):
    img = Image.open(file_path)
    img_tk = ImageTk.PhotoImage(img)
    picture = Label(root, image=img_tk)
    picture.grid(row=2, column=0)
    picture.image = img_tk

root = tk.Tk()
root.title("H.U.M.M.U.S.")
root.geometry("720x720")

text1 = Label(root, text="Press the Button to Process an Image.", font=("Arial", 20))
text1.grid(row=0, column=0)

style = ttk.Style()
style.configure('TButton', font=('Arial', 14), foreground='black', background='#4CAF50', padding=10)

upload_button = ttk.Button(root, text="Upload Image", command=upload_wrapper, style='TButton')
upload_button.grid(row=1, column=0)

butt_display = tk.Button(root, text="Display Image", command=lambda: showimage(file_path))

root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()