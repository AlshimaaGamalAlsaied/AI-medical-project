from flask import Flask, render_template, request, jsonify, flash, redirect  # Import jsonify
from functions import process_image

import os

app = Flask(__name__)

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            output_filename = process_image(file.filename)
            
            return render_template('home.html', output_image=output_filename)
    
    return render_template('home.html', output_image=None)

@app.route('/process_image/<input_filename>')
def process_image_route(input_filename):
    output_filename = process_image(input_filename)
    result_info = "Result information goes here..."
    
    # Assuming UPLOAD_FOLDER is '/data/nih_new/images-small'
    output_image_url = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    
    return jsonify({'output_image': output_image_url, 'result_info': result_info})

if __name__ == '__main__':
    app.run(debug=True)