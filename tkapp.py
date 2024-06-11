import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
from sklearn.datasets import load_iris
from PIL import Image, ImageTk

# Load the model and scaler
model = joblib.load('./model/logistic_regression_model.pkl')
scaler = joblib.load('./model/scaler.pkl')

# Load Iris dataset for reference
iris = load_iris()

# Initialize the main window
root = tk.Tk()
root.title("Iris Flower Species Prediction")

# # Load and display an image 
# image = Image.open('./asset/iris_petal-sepal.png')
# image = ImageTk.PhotoImage(image)

# # Create a label to display the image
# image_label = tk.Label(root, image=image)
# image_label.pack()

# Create and place labels and input fields
tk.Label(root, text="Sepal Length").grid(row=0, column=0)
sepal_length = tk.Entry(root)
sepal_length.grid(row=0, column=1)

tk.Label(root, text="Sepal Width").grid(row=1, column=0)
sepal_width = tk.Entry(root)
sepal_width.grid(row=1, column=1)

tk.Label(root, text="Petal Length").grid(row=2, column=0)
petal_length = tk.Entry(root)
petal_length.grid(row=2, column=1)

tk.Label(root, text="Petal Width").grid(row=3, column=0)
petal_width = tk.Entry(root)
petal_width.grid(row=3, column=1)

# Define the prediction function
def predict():
    try:
        # Collect input data
        features = np.array([[float(sepal_length.get()), float(sepal_width.get()), float(petal_length.get()), float(petal_width.get())]])
        # Scale the input data
        features = scaler.transform(features)
        # Predict the species
        prediction = model.predict(features)
        # Display the result
        species = iris.target_names[prediction][0]
        messagebox.showinfo("Prediction", f"The predicted species is: {species}")
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numeric values.")

# Create and place the predict button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=4, columnspan=2)

# Run the Tkinter event loop
root.mainloop()
