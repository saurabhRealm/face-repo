import os
import uuid
import cv2
import face_recognition
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, Response, send_file

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize known faces
known_face_encodings = []
known_face_names = []
users = []

# Load known faces
if os.path.exists('users.txt'):
    with open('users.txt', 'r') as f:
        users = [line.strip().split(',') for line in f]
        for name, filename in users:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    encoding = encodings[0]
                    known_face_encodings.append(encoding)
                    known_face_names.append(name)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image = request.files.get('image')
        capture = request.form.get('capture')
        errors = {}

        # Validation for name
        if not name:
            errors['name'] = "Name is required!"

        # Validation for image or capture
        if not image and not capture:
            errors['image'] = "You must upload an image or capture one!"

        if errors:
            return render_template('register.html', errors=errors)

        # If there's an image, save it
        if image:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
        elif capture:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Uncomment and implement the following function if capturing from camera
            # save_image_from_camera(image_path)

        # Save user data to a file
        with open('users.txt', 'a') as f:
            f.write(f"{name},{filename}\n")
        
        # Process face recognition
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(name)
        
        return redirect(url_for('home'))

    return render_template('register.html')


@app.route('/view_users', methods=['GET', 'POST'])
def view_users():
    users = []
    if os.path.exists('users.txt'):
        with open('users.txt', 'r') as f:
            users = [line.strip().split(',') for line in f]
    
    if request.method == 'POST':
        delete_name = request.form.get('delete_user')
        if delete_name:
            with open('users.txt', 'r') as f:
                lines = f.readlines()
            
            with open('users.txt', 'w') as f:
                for line in lines:
                    if not line.startswith(delete_name + ','):
                        f.write(line)
            
            image_to_delete = None
            for name, image in users:
                if name == delete_name:
                    image_to_delete = image
                    break
            
            if image_to_delete:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image_to_delete))
            
            known_face_encodings.clear()
            known_face_names.clear()
            if os.path.exists('users.txt'):
                with open('users.txt', 'r') as f:
                    users = [line.strip().split(',') for line in f]
                    for name, filename in users:
                        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        if os.path.exists(image_path):
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            if encodings:
                                encoding = encodings[0]
                                known_face_encodings.append(encoding)
                                known_face_names.append(name)
            
            return redirect(url_for('view_users'))
    
    return render_template('view_users.html', users=users)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        image = request.files.get('image')

        if not image:
            return render_template('attendance.html', result="No image provided!", error=True)

        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        try:
            uploaded_image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(uploaded_image)

            if encodings:
                encoding = encodings[0]
                matches = face_recognition.compare_faces(known_face_encodings, encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    timestamp = pd.Timestamp.now()

                    # Load existing attendance records
                    if os.path.exists('attendance_records.xlsx'):
                        df = pd.read_excel('attendance_records.xlsx')
                    else:
                        df = pd.DataFrame(columns=['User', 'Timestamp'])

                    # Add new record
                    new_record = pd.DataFrame({'User': [name], 'Timestamp': [timestamp]})
                    df = pd.concat([df, new_record], ignore_index=True)
                    df.to_excel('attendance_records.xlsx', index=False)

                    result = f"Attendance marked for {name}."
                else:
                    result = "No matching user found."
            else:
                result = "No face detected in the image."

        except Exception as e:
            result = f"Error processing image: {str(e)}"
            print(f"Error: {e}")

        return render_template('attendance.html', result=result)

    return render_template('attendance.html')

@app.route('/export_attendance')
def export_attendance():
    if not os.path.exists('attendance_records.xlsx'):
        return "No attendance records available!", 404

    df = pd.read_excel('attendance_records.xlsx')
    output = 'attendance_records.xlsx'

    return send_file(output, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
