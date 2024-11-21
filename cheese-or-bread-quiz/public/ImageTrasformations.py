from PIL import Image
from PIL import ImageShow
from flask import Flask, request, jsonify
from os import mkdir, getcwd

app = Flask(__name__)

def get_rotations(img, file_path, folder, file_count):
    img_1 = img.rotate(90, expand=True)
    img_1.save(f"{file_path}/{folder}_{file_count}.png")
    img_2 = img.rotate(180, expand=True)
    img_2.save(f"{file_path}/{folder}_{file_count + 1}.png")
    img_3 = img.rotate(270, expand=True)
    img_3.save(f"{file_path}/{folder}_{file_count + 2}.png")

@app.route('/image_transforms', methods=['POST'])
def transform_image():
    data = request.json
    img_file = data["img_file"]
    user_answer = data["user_answer"]
    img_path = getcwd() + "/cheese-or-bread-quiz/public/" + img_file
    with Image.open(img_path) as img:
        file_count = 1
        folder = img_file.split('.')[0].split("/")[2]
        file_path = getcwd() + "/cheese-or-bread-quiz/public/" + "/".join(img_file.split("/")[:-1]) + "/" + folder
        folder = img_file.split('.')[0].split("/")[2]
        file_name = img_file.split("/")[2]
        
        try:
            mkdir(f"{file_path}")
        except:
            pass
        img.save(f"{file_path}/{file_name}")
        get_rotations(img, file_path, folder, file_count)
        file_count += 3
        
        flipped_lr_img = img.transpose(0)
        flipped_lr_img.save(f"{file_path}/{folder}_{file_count}.png")
        file_count += 1
        
        get_rotations(flipped_lr_img, file_path, folder, file_count)
        file_count +=3
        return jsonify({"folder": f"{file_path}"})
        

if __name__ == '__main__':
    app.run(debug=True)
    
