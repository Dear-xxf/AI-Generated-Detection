from flask import Flask, request, render_template, jsonify
import os
from GoogleNetV3 import GoogleNetV3Classifier

app = Flask(__name__)

# 确保上传的图片保存在这个目录下
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = GoogleNetV3Classifier('./dataset')
model.load_model('ai_generated_detection.pth')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload/img', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # 保存文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        prediction = model.predict(filename)
        return jsonify({'message': 'File uploaded successfully!', 'value': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5001)