from flask import Flask, render_template, request
from main import save_from_video
import os 
import time
import json
import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='P@ssw0rd',
    database="mydatabase"
)
mycursor = mydb.cursor()

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('index.html')

@app.route('/user-home.html')
def user_section():
    return render_template('user-home.html')

@app.route('/admin-home.html')
def admin_section():
    return render_template('admin-home.html')

@app.route('/vid_upload', methods=['POST'])
def admin_upload_vid():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            file_path = os.path.join('instance', 'uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
            save_from_video(file_path)
    return render_template('admin-home.html', return_message='from_vid_upload')

@app.route('/db_refresh')
def display_rec_db():
    mycursor.execute('select id, plate_number, loaction, date, time from recognized_vehicle')
    result = json.dumps(mycursor.fetchall())
    return render_template('admin-home.html', return_message='from_db_display', db_value=result)



if __name__ == '__main__':
    app.run(debug=True)
