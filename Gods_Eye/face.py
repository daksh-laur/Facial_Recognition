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

    # start
    known_face_encodings = []
    known_face_names = []

    cur_direc = os.getcwd()
    path = os.path.join(cur_direc, 'upload/')
    # list_of_files = [f for f in glob.glob(path+'*.jpg')]
    for i in glob.glob(path):
        list_of_files = [f for f in glob.glob(i+'*.jpg')]


    for i in range(len(list_of_files)):
        x = face_recognition.load_image_file(list_of_files[i])
        y = face_recognition.face_encodings(x)[0]
        known_face_encodings.append(y)
        k=os.path.basename(list_of_files[i])
        known_face_names.append(k[:-4])




    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    

        # end 

    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "UNKNOWN"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                for i in face_names:
                    if i.upper() not in all_names:
                        if i!="UNKNOWN":
                            all_names.append(i.upper())
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/', methods=['GET',"POST"])
def home():
    return render_template('final_homepage.html')



@app.route('/upload', methods=['GET',"POST"])
def upload():
    if request.method == 'POST':

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return render_template('final_upload_page.html')


@app.route('/temp',methods=['GET',"POST"])
def temp():
    return render_template('final_face_recog.html')

@app.route('/video', methods=['GET',"POST"])
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/names', methods=['GET',"POST"])
def names():
    
    return render_template('final_display.html',list=all_names)



if __name__=='__main__':
    app.run(debug=True)

