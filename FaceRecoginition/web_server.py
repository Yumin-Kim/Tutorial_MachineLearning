# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect ,send_file
import numpy as np
import math

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

test01_image = face_recognition.load_image_file("test01.jpg")
test01_face_encoding = face_recognition.face_encodings(test01_image)[0]

# Load a sample picture and learn how to recognize it.
test02_image = face_recognition.load_image_file("test02.jpg")
test02_face_encoding = face_recognition.face_encodings(test02_image)[0]

test03_image = face_recognition.load_image_file("test03.jpg")
test03_face_encoding = face_recognition.face_encodings(test03_image)[0]

# Load a sample picture and learn how to recognize it.
test04_image = face_recognition.load_image_file("test04.jpg")
test04_face_encoding = face_recognition.face_encodings(test04_image)[0]

test05_image = face_recognition.load_image_file("test05.jpg")
test05_face_encoding = face_recognition.face_encodings(test05_image)[0]

# Load a sample picture and learn how to recognize it.
test06_image = face_recognition.load_image_file("test06.jpg")
test06_face_locations = face_recognition.face_locations(test06_image, number_of_times_to_upsample=0, model="cnn")
test06_face_encoding = face_recognition.face_encodings(test06_image,test06_face_locations)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    test01_face_encoding,
    test02_face_encoding,
    test03_face_encoding,
    test04_face_encoding,
    test05_face_encoding,
    test06_face_encoding,
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "person1",
    "person2",
    "person3",
    "person4",
    "person5",
    "person6",
]

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return compare_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Is this a picture of Obama?</title>
    <h1>Upload a picture and see if it's a picture of Obama!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''
@app.route('/compare', methods=['POST'])
def compare_test():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        print(request.files)
        file = request.files['image']

        print("Test compare Logic")    
        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            print('Learned encoding for', len(known_face_encodings), 'images.')
            unknown_image = face_recognition.load_image_file(file)
            face_locations = face_recognition.face_locations(unknown_image)
            face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

            test = "진행중"
            face_found = False
            is_obama = False
            face_names = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
            # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
            #해당 이름 변경 하는 코드
                if matches[best_match_index]:
                    print(face_distances[best_match_index])
                    test = "name : " + known_face_names[best_match_index] +" | per : " +str(math.ceil(1-face_distances[best_match_index])*100) +"%"

            result = {
                "face_found_in_image": test,
                "is_picture_of_obama": face_names
            }
        print(result)
        return jsonify(result)

@app.route("/image")
def RequestImage():
    if request.method == "GET":
        return send_file("test01.jpg",mimetype="image/jpg")
@app.route("/json")
def RequsetJSON():
    if request.method == "GET":
        result = {
        "face_found_in_image": "face_found",
        "is_picture_of_obama": "is_obama"
        }
        return jsonify(result)
# 현재 실시간 영상을 처리하는 로직보다는 캡쳐한 이미지를 http프로토콜 통신을 통해서 처리한다음 json데이터로 전송하는 로직 구현
# 아래 route가 그 역할을 한다. 
# @app.route("/compare")


def detect_faces_in_image(file_stream):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    known_face_encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
                            0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
                            0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
                            0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                            0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
                           -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
                           -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
                           -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
                           -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
                            0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
                            0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
                           -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
                            0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
                            0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
                            0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
                            0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
                           -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
                           -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
                           -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
                            0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
                            0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
                            0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
                           -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
                            0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
                           -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
                            0.07417042,  0.07126575,  0.00209804]

    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    face_found = False
    is_obama = False

    if len(unknown_face_encodings) > 0:
        face_found = True
        # See if the first face in the uploaded image matches the known face of Obama
        match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
        if match_results[0]:
            is_obama = True

    # Return the result as json
    result = {
        "face_found_in_image": face_found,
        "is_picture_of_obama": is_obama
    }
    return jsonify(result)

def compare_image(file_stream):
    print('Learned encoding for', len(known_face_encodings), 'images.')
    unknown_image = face_recognition.load_image_file(file_stream)
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    
    face_found = False
    is_obama = False
    face_names = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.5)
    # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
    #해당 이름 변경 하는 코드
        if matches[best_match_index]:
            print(face_distances[best_match_index])
            test = "name : " + known_face_names[best_match_index] +" | per : " +str(math.ceil(1-face_distances[best_match_index])*100) +"%"

    result = {
        "face_found_in_image": test,
        "is_picture_of_obama": face_names
    }
    print(result)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)