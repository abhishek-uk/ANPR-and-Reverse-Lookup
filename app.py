from flask import Flask, render_template, request
from main import save_plates_from_video
import os 
import time

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
            # save_plates_from_video(file_path)
            time.sleep(2)
        
    return render_template('admin-home.html', return_message='from_vid_upload')




if __name__ == '__main__':
    app.run(debug=True)
