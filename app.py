from flask import Flask, render_template, Response, request, redirect, url_for, flash, session, send_file
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import cv2
import face_recognition
import numpy as np
import threading
import os
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///face.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(db.Model, UserMixin): 
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, nullable=False, unique=True)
    password = db.Column(db.String, nullable=False)

class Person(db.Model):
    __tablename__ = 'person'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    position = db.Column(db.String, nullable=True)
    description = db.Column(db.String, nullable=True)
    image = db.Column(db.String, nullable=True)
    status = db.Column(db.String, nullable=False, default='unverified')  # New status field

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials, please try again.", "danger")
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken. Please choose a different one.", "danger")
            return redirect(url_for('signup'))

        try:   
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            new_user = User(username=username, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()  # Rollback to prevent PendingRollbackError
            flash("An error occurred. Please try again.", "danger")
            print(f"Error: {e}")  # Log the error for debugging
 
    return render_template('signup.html')
  
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/add_user', methods=['GET', 'POST'])
@login_required
def add_user():
    if current_user.username != "Admin":
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('User added successfully!')
        return redirect(url_for('index'))

    return render_template('add_user.html')

# OpenCV video capture
camera = cv2.VideoCapture(0)

# Reduce frame size for faster processing
def resize_frame(frame, scale=0.25):
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width * scale), int(height * scale)))

known_faces = []
@app.before_request
def load_known_faces_on_startup():
    load_known_faces()

# Function to load known faces
def load_known_faces():
    global known_faces
    known_faces = []
    people = Person.query.all()
    for person in people:
        if person.image and os.path.exists(person.image):
            image = face_recognition.load_image_file(person.image)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_faces.append({
                    "name": person.name,
                    "age": person.age,
                    "position": person.position,
                    "description": person.description,
                    "status": person.status,  # Include status
                    "encoding": encoding[0]
                })

face_encodings_cache = []
face_locations = []
lock = threading.Lock()

def process_frame(frame):
    global face_encodings_cache, face_locations
    small_frame = resize_frame(frame)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, locations)
    with lock:
        face_locations = locations
        face_encodings_cache = encodings

def generate_frames():

    camera = cv2.VideoCapture(0)
    
    frame_count = 0
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break
            if frame_count % 5 == 0:
                threading.Thread(target=process_frame, args=(frame,)).start()
            with lock:
                for face_encoding, (top, right, bottom, left) in zip(face_encodings_cache, face_locations):
                    matches = face_recognition.compare_faces([f["encoding"] for f in known_faces], face_encoding)
                    name, age, position, description, status = "Unknown", "", "", "", "unverified"
                    face_distances = face_recognition.face_distance([f["encoding"] for f in known_faces], face_encoding)
                    if face_distances.size > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            user = known_faces[best_match_index]
                            name, age, position, description = user["name"], user["age"], user["position"], user["description"]
                            status = user["status"]
                    top, right, bottom, left = [int(coord / 0.25) for coord in (top, right, bottom, left)]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"{name}, {age}, {position}, Status: {status}"
                    cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, description, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            frame_count += 1

        except Exception as e:
            print(e) 
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_person', methods=['GET', 'POST']) 
def add_person(): 
    print("here")
    if request.method == 'POST':
        print("here")
        name = request.form['name']
        age = request.form['age']
        position = request.form['position']
        description = request.form['description']
        status = request.form['status']
         
        # Handle file upload
        image_file = request.files['image'] 
        image_path = None   
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

        new_person = Person(name=name, age=age, position=position, description=description, image=image_path, status = status)
        db.session.add(new_person)
        db.session.commit()

        load_known_faces()  # Reload known faces dynamically
        print("here")
        return redirect(url_for('index'))

    people = Person.query.all()
    print("this is here")
    return render_template('add_person.html', people=people)

@app.route('/add_person_user', methods=['GET', 'POST'])  
def add_person_user():
    if request.method == 'POST': 
        name = request.form['name']
        age = request.form['age']
        position = request.form['position']
        description = request.form['description']
        status = request.form['status']  

        # Handle file upload
        image_file = request.files['image']
        image_path = None
        if image_file and image_file.filename != '':
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

        new_person = Person(name=name, age=age, position=position, description=description, image=image_path , status = status)
        db.session.add(new_person)
        db.session.commit()

        load_known_faces()  # Reload known faces dynamically

        return redirect(url_for('index')) 

    people = Person.query.all()
    return render_template('add_person_user.html', people=people)

@app.route('/delete_person/<int:person_id>', methods=['POST'])
def delete_person(person_id):
    person = Person.query.get(person_id)
    if person:
        if person.image and os.path.exists(person.image):
            os.remove(person.image)
        db.session.delete(person)
        db.session.commit() 

        load_known_faces()  # Reload known faces dynamically

    return redirect(url_for('add_person'))

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    if 'image' not in request.files:
        flash("No file part", "danger")
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        flash("No selected file", "danger")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    image = cv2.imread(file_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, locations)

    for encoding, (top, right, bottom, left) in zip(encodings, locations):
        matches = face_recognition.compare_faces([f["encoding"] for f in known_faces], encoding)
        name, age, position, description, status = "Unknown", "", "", "", "unverified"
        
        face_distances = face_recognition.face_distance([f["encoding"] for f in known_faces], encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                user = known_faces[best_match_index]
                name, age, position, description = user["name"], user["age"], user["position"], user["description"]
                status = user["status"]
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name}, {age}, {position}, Status: {status}"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, description, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(processed_path, image)
    return send_file(processed_path, mimetype='image/jpeg')


with app.app_context():
    load_known_faces()

@app.route('/users')
@login_required
def users():
    if current_user.username != "Admin":
        return redirect(url_for('index'))
    all_users = User.query.all()
    return render_template('users.html', users=all_users)

@app.route('/edit_user/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_user(id):
    if current_user.username != "Admin":
        return redirect(url_for('index'))

    user = User.query.get_or_404(id)

    if request.method == 'POST':
        user.username = request.form['username']
        password = request.form['password']
        if password:
            user.password = generate_password_hash(password, method='pbkdf2:sha256')

        db.session.commit()

        flash('User updated successfully!')
        return redirect(url_for('users'))

    return render_template('edit_user.html', user=user)

@app.route('/delete_user/<int:id>')
@login_required
def delete_user(id):
    if current_user.username != "Admin":
        return redirect(url_for('index'))

    user = User.query.get_or_404(id)
    db.session.delete(user)
    db.session.commit()

    flash('User deleted successfully!')
    return redirect(url_for('users'))

@app.route('/')
@login_required
def index():
    if current_user.username == "Admin":
        return render_template('adminindex.html')
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
