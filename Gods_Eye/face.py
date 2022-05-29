from flask import Flask, redirect, render_template, Response, url_for,request
import cv2
import face_recognition
import numpy as np
import os
import glob
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired

app=Flask(__name__)
camera = cv2.VideoCapture(0)


app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'upload'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

all_names=[]

def gen_frames():  

    
    uploaded_facial_encodings = []
    uploaded_face_names = []

    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'upload/')
    
    for i in glob.glob(path):
        list_of_files = [f for f in glob.glob(i+'*.jpg')]


    for i in range(len(list_of_files)):
        x = face_recognition.load_image_file(list_of_files[i])
        y = face_recognition.face_encodings(x)[0]
        uploaded_facial_encodings.append(y)
        k=os.path.basename(list_of_files[i])
        uploaded_face_names.append(k[:-4])




    face_detected = []
    facial_encodings = []
    face_names = []
    process_this_frame = True
    

        

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video t
            frame_small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image fto RGB color
            rgb_small = frame_small[:, :, ::-1]

           
            # Find   faces and current frame of video
            face_detected = face_recognition.face_locations(rgb_small)
            facial_encodings = face_recognition.face_encodings(rgb_small, face_detected)
            face_names = []
            
            for i in facial_encodings:
                # comparing detected faces with uploaded database
                matches = face_recognition.compare_faces(uploaded_facial_encodings, i)
                name = "UNKNOWN" #default name as UNKNOWN
                
                face_distances = face_recognition.face_distance(uploaded_facial_encodings, i)
                best_match = np.argmin(face_distances)
                # setting the best match name to name variable 
                if matches[best_match]:
                    name = uploaded_face_names[best_match]

                # noting the detected criminal names 
                face_names.append(name)
                for i in face_names:
                    if i.upper() not in all_names:
                        if i!="UNKNOWN":
                            all_names.append(i.upper())
            

            
            for (top, right, bottom, left), name in zip(face_detected, face_names):
                # resetting the frame size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # making frame over detected faces
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # print names under the frame
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Starting of the webpage

@app.route('/', methods=['GET',"POST"]) # displaying the homepage 
def home():
    return render_template('final_homepage.html')


# The page to upload criminal database and store the data locally with us. 
@app.route('/upload', methods=['GET',"POST"])
def upload():
    if request.method == 'POST':

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('final_upload_page.html')

# face recognition portal 
@app.route('/temp',methods=['GET',"POST"])
def temp():
    return render_template('final_face_recog.html')

@app.route('/video', methods=['GET',"POST"])
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# displaying the names of detected criminals

@app.route('/names', methods=['GET',"POST"])
def names():
    
    return render_template('final_display.html',list=all_names)



if __name__=='__main__':
    app.run(debug=True)

