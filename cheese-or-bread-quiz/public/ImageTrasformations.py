from PIL import Image
from PIL import ImageShow
from flask import Flask, request, jsonify
from os import mkdir, getcwd
from jimbo_brain import CHEESE, BREAD, train_model, model_pipeline, jimboMenu

app = Flask(__name__)

def get_rotations(img, file_path, folder, file_count, answers, user_answer):
    img_1 = img.rotate(90, expand=True)
    train_model(model_pipeline, answers[user_answer.lower()], img_1)
    img_2 = img.rotate(180, expand=True)
    train_model(model_pipeline, answers[user_answer.lower()], img_2)
    img_3 = img.rotate(270, expand=True)
    train_model(model_pipeline, answers[user_answer.lower()], img_3)

@app.route('/image_transforms', methods=['POST'])
def transform_image():
    data = request.json
    img_file = data["img_file"]
    user_answer = data["user_answer"]
    answers = {"cheese": CHEESE, "bread": BREAD}
    img_path = "C:/Users/swerv/OneDrive/Desktop/Senior_Design_2/cheese-or-bread-quiz/public/" + img_file
    with Image.open(img_path) as img:
        file_count = 1
        folder = img_file.split('.')[0].split("/")[2]
        file_path = "C:/Users/swerv/OneDrive/Desktop/Senior_Design_2/cheese-or-bread-quiz/public/" + "/".join(img_file.split("/")[:-1]) + "/" + folder
        folder = img_file.split('.')[0].split("/")[2]
        file_name = img_file.split("/")[2]
        
        try:
            mkdir(f"{file_path}")
        except:
            pass
        train_model(model_pipeline, answers[user_answer.lower()], img)
        get_rotations(img, file_path, folder, file_count, answers, user_answer)
        file_count += 3
        
        flipped_lr_img = img.transpose(0)
        train_model(model_pipeline, answers[user_answer.lower()], flipped_lr_img)
        file_count += 1
        
        get_rotations(flipped_lr_img, file_path, folder, file_count, answers, user_answer)
        file_count +=3
        return jsonify({"folder": f"{file_path}"})
        

if __name__ == '__main__':
    app.run(debug=True)
    
