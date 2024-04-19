from flask import Flask, render_template, request
# from main import save_plates_from_video
import os 

app = Flask(__name__)

#Defin the root to render the html page 
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to handle file upload 








if __name__ == '__main__':
    print('started progajflsjfl')
    app.run(debug=True)
