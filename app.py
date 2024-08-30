import os
import uuid
import re
import base64
import cv2
import face_recognition
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, Response, send_file, session, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = os.urandom(24).hex()  # Set the secret key

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize known faces
known_face_encodings = []
known_face_names = []
users = []
user_details = []

def load_known_faces():
    # Example: Load known faces from files and encode them
    # known_image = face_recognition.load_image_file("path/to/known_face.jpg")
    # known_face_encoding = face_recognition.face_encodings(known_image)[0]
    # known_face_encodings.append(known_face_encoding)
    # known_face_names.append("Name of Person")
    pass

load_known_faces()
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
    success_message = session.pop('success_message', None)
    return render_template('home.html', success_message=success_message)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def save_image_from_camera(image_path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Please check if the camera is connected and accessible.")
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture image. Please try again.")
    
    cv2.imwrite(image_path, frame)
    cap.release()
    cv2.destroyAllWindows()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        image = request.files.get('image')
        capture_image_data = request.form.get('camera_image_data')
        errors = {}

        # Validation for name
        if not name:
            errors['name'] = "Name is required!"

        # Validation for image or capture
        if not image and not capture_image_data:
            errors['image'] = "You must upload an image or capture one!"

        if errors:
            return render_template('register.html', errors=errors)

        filename = None

        # If there's an image, save it
        if image:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
        elif capture_image_data:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Decode and save the captured image
            image_data = re.sub('^data:image/jpeg;base64,', '', capture_image_data)
            image_data = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_data))
            image.save(image_path)

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
        
        # Set success message in session
        session['success_message'] = 'Registration successful! Redirecting to home...'
        return redirect(url_for('home'))

    return render_template('register.html')
@app.route('/view_users', methods=['GET', 'POST'])
def view_users():
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
        if 'image' not in request.files:
            return jsonify(result="No image provided."), 400

        image_file = request.files['image']

        if image_file and allowed_file(image_file.filename):
            try:
                # Read and process the uploaded image
                image = Image.open(image_file.stream)
                image_np = np.array(image)
                
                # Find all face encodings in the uploaded image
                face_locations = face_recognition.face_locations(image_np)
                face_encodings = face_recognition.face_encodings(image_np, face_locations)
                
                user_detected = False
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    if True in matches:
                        user_detected = True
                        break

                if user_detected:
                    return jsonify(result="User detected successfully.", success=True), 200
                else:
                    return jsonify(result="User not detected.", success=False), 404
                
            except Exception as e:
                print(f"Error processing image: {e}")
                return jsonify(result="Error processing image."), 500

            return jsonify(result="Invalid image.", success=False), 404

    # Render the attendance page for GET requests
    return render_template('attendance.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']

@app.route('/export_attendance')
def export_attendance():
    if not os.path.exists('attendance_records.xlsx'):
        return "No attendance records available!", 404

    df = pd.read_excel('attendance_records.xlsx')
    output = 'attendance_records.xlsx'

    return send_file(output, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
