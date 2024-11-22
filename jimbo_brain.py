print("Importing libraries...")

from river import metrics
from river.base.typing import RegTarget
from deep_river.regression import Regressor
from deep_river.utils.tensor_conversion import float2tensor
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
import threading
from threading import Thread
import ImageTransformations
from flask import Flask, request

print("Finished")

PATH = "model_weights.pth"
CHEESE = 0
BREAD = 1

trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=128, antialias=True)])
data_lock = threading.Lock()

class MyModule(nn.Module):
    def __init__(self, n_features: int):
        super(MyModule, self).__init__()

        print("n_features: ", n_features)

        self.conv1 = nn.Conv2d(in_channels=n_features, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(3721, 128) #3721
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)

        #x = x.view(-1, 64*12*12)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x

def predict_one_tensor(regressor: Regressor, x: torch.Tensor) -> RegTarget:
    if not regressor.module_initialized:
        regressor.kwargs["n_features"] = x.dim()
        regressor.initialize_module(**regressor.kwargs)
    regressor.module.eval()
    with torch.inference_mode():
        y_pred = regressor.module(x).item()
    return y_pred

def train_model(model: Regressor, metric: metrics.MAE, guess: int, image: Image.Image):
    x = image.resize((128, 128))

    y_ten = float2tensor(guess)
    x_ten = trans(x)

    y_pred = predict_one_tensor(model, x_ten)
    print (y_pred)
    metric.update(y_true=y, y_pred=y_pred)
    model._learn(x=x_ten, y=y_ten)

def predict_model(model, image: Image.Image) -> RegTarget:
    x = image.resize((128, 128))
    x = trans(x)
    return predict_one_tensor(model, x)

#cheese or bread
def cob(val: float) -> str:
    return "cheese" if val < 0.5 else "bread"

input("Waiting to continue...")
        
def run_flask_app(model: Regressor, metric: metrics.MAE):
    app = Flask(__name__)

    @app.route('/image_transforms', methods=['POST'])
    def train_model_on_answer():
        data = request.json
        img_file = data["img_file"]
        user_answer = data["user_answer"]

        print(img_file, user_answer)

        if user_answer.lower() == "cheese":
            guess = CHEESE
        elif user_answer.lower() == "bread":
            guess = BREAD
        else:
            return

        with data_lock:
            for image in ImageTransformations.retrieve_images(img_file):
                train_model(model, metric, guess, image)

    app.run(debug=False)


def user_interface():
    #tweak lr if need be, i've tested up to 0.01, seems to work at values around 0.005
    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"

    if use_cuda:
        print("Starting model on the GPU.\n")
    else:
        print("Starting model on the CPU.\n")

    model_pipeline = Regressor(module=MyModule, loss_fn='mse', lr=0.01, optimizer_fn='sgd', device=device_str)
    metric = metrics.MAE()

    while True:
        inp = input("""\
Enter a choice:
    1) Try a test prediction
    2) Close
    3) Save model
    4) Load model
    5) Start Flask App
    6) Print metrics
                    
> """)

        if inp == '1':
            def predict_and_analyze(image_path, true_val):
                with data_lock:
                    image = Image.open(image_path).convert("RGB")
                    y_pred = predict_model(model_pipeline, image)
                    print("Jimbo predicted:", "%.2f"%(y_pred) + ", the correct answer was:", true_val)
                    print("Jimbo was off by ", "%.2f"%abs(y_pred - true_val), ".")
                    word_true = cob(true_val)
                    word_guess = cob(y_pred)
                    outcome = "CORRECT!" if word_guess == word_true else "WRONG!"
                    print("Jimbo thinks this is a picture of", word_guess + ".", outcome)

            predict_and_analyze("cheese-or-bread-quiz\\data\\cheese.png", CHEESE)
            predict_and_analyze("cheese-or-bread-quiz\\data\\bread.png", BREAD)
            
        elif inp == '2':
            print("Closing...")
            exit()

        elif inp == '3':
            print("Saving...")
            with data_lock:
                with open(PATH, 'wb+') as f:
                    pickle.dump(model_pipeline, f)

        elif inp == '4':
            print("Loading...")
            with data_lock:
                model_pipeline = pickle.load(open(PATH,"rb"))

        elif inp == '5':
            print("Starting flask app...")
            thread = Thread(target=run_flask_app, args=(model_pipeline, metric))
            thread.start()

        elif inp == '6':
            print(f'MAE: {metric.get():.2f}')

if __name__ == "__main__":
    user_interface()
