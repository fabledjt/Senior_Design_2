from PIL import Image
from PIL import ImageShow
from flask import Flask, request, jsonify

app = Flask(__name__)


img_file = "Cheese09.png"

def get_rotations(img, file_name, file_count):
    img_1 = img.rotate(90, expand=True)
    img_1.save(f"{file_name.split('.')[0]}_{file_count}.png")
    img_2 = img.rotate(180, expand=True)
    img_2.save(f"{file_name.split('.')[0]}_{file_count + 1}.png")
    img_3 = img.rotate(270, expand=True)
    img_3.save(f"{file_name.split('.')[0]}_{file_count + 2}.png")

@app.route('/image_transforms', methods['POST'])
def transform_image(img_file):
    with Image.open(img_file) as img:
        file_count = 1
        get_rotations(img, img_file, file_count)
        file_count += 3
        
        flipped_lr_img = img.transpose(0)
        flipped_lr_img.save(f"{img_file.split('.')[0]}_{file_count}.png")
        file_count += 1
        
        get_rotations(flipped_lr_img, img_file, file_count)
        file_count +=3

if __name__ == '__main__':
    app.run(debug=True)
    
