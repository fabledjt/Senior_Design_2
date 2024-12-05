from PIL import Image
from PIL import ImageShow
from os import getcwd

def get_rotations(img: Image.Image) -> list:
    imgs = []
    imgs.append(img)
    imgs.append(img.rotate(90, expand=True))
    imgs.append(img.rotate(180, expand=True))
    imgs.append(img.rotate(270, expand=True))
    
    flipped_lr_img = img.transpose(0)
    imgs.append(flipped_lr_img)
    imgs.append(flipped_lr_img.rotate(90, expand=True))
    imgs.append(flipped_lr_img.rotate(180, expand=True))
    imgs.append(flipped_lr_img.rotate(270, expand=True))

    return imgs

def retrieve_images(img_file: str) -> list:
    img_path = "C:/Users/swerv/OneDrive/Desktop/Senior_Design_2/cheese-or-bread-quiz/public" + img_file
    with Image.open(img_path).convert("RGB") as img:
        return get_rotations(img)
 

# if __name__ == '__main__':
#     app.run(debug=True)
    
