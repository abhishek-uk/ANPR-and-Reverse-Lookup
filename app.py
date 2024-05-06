from flask import Flask, render_template, request
from main import save_from_video, predict_hsrp
import os 
import time
import json
import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user='root',
    password='P@ssw0rd',
    database="nprarl_db"
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
            # file_path = os.path.join('instance', 'uploads', uploaded_file.filename)
            file_path = os.path.join('static', 'uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
            save_from_video(file_path)
    return render_template('admin-home.html', return_message='from_vid_upload')

@app.route('/db_refresh')
def display_rec_db():
    mycursor.execute('select id, plate_number, loaction, date, time from recognized_vehicles')
    result = json.dumps(mycursor.fetchall())
    return render_template('admin-home.html', return_message='from_db_display', db_value=result)


@app.route('/search/<number>')
def search_plate(number):
    mycursor.execute(f"select plate_number, loaction, hsrp, date, time, video_file_name, frame_time, plate_img_name from recognized_vehicles where plate_number = '{number}';")
    result = mycursor.fetchall()

    if len(result) == 0:
        return render_template('user-home.html',
                                plate_not_found_section='''
                                                    <div id="plate-not-found"> 
                                                        <h1>The Plate is not Found</h1>
                                                    </div>
                                '''
                )

    plate_number = result[0][0]
    mycursor.execute(f'select if (exists(select * from black_listed_vehicles where plate_number = "{plate_number}"), True, False)')
    black_listed = mycursor.fetchone()
    if black_listed[0] == True:
        BL = 'Yes'
    else:
        BL = 'No'

    plate_img_path = os.path.join('static', 'detected_plates', result[0][7])
    hrsp = predict_hsrp(plate_img_path)
    print(hrsp)
    
    return render_template('user-home.html',
                           plate_found_section=f'''
                                            <div id="plate-found">
                                                <div class="plate-details">
                                                    <div class="grid-item" id="dis-plate-no">Plate Number: { result[0][0] }</div>
                                                    <div class="grid-item" id="dis-location">Location: { result[0][1] }</div>
                                                    <div class="grid-item" id="dis-hsrp">HSRP: { result[0][2] }</div>
                                                    <div class="grid-item" id="dis-date">Date: { result[0][3] }</div>
                                                    <div class="grid-item" id="dis-time">Time: { result[0][4] }</div>
                                                    <div class="grid-item" id="dis-black-listed">Black listed: { BL }</div>
                                                </div>
                                            </div>
                                                ''',
                            display_variable='vid_db', 
                            vid_file_name = result[0][5],
                            start_time = result[0][6]
            )



if __name__ == '__main__':
    app.run(debug=True)
