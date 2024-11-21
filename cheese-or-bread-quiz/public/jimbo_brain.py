print("Importing libraries...")

from river import metrics, compose, datasets
from river.base.typing import RegTarget
from deep_river.regression import Regressor
from deep_river.utils.tensor_conversion import float2tensor
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image

print("Finished")

PATH = "model_weights.pth"
CHEESE = 0
BREAD = 1

class MyModule(nn.Module):
    def __init__(self, n_features):
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

def train_model(model, guess: int, image: Image):
    x = image.resize((128, 128))

    y_ten = float2tensor(guess)
    x_ten = trans(x)

    y_pred = predict_one_tensor(model, x_ten)
    print (y_pred)
    metric.update(y_true=y, y_pred=y_pred)
    model._learn(x=x_ten, y=y_ten)

def predict_model(model, image: Image):
    x = image.resize((128, 128))
    x = trans(x)
    return predict_one_tensor(model, x)

#cheese or bread
def cob(val):
    "cheese" if y < 0.5 else "bread"

    
#tweak lr if need be, i've tested up to 0.01, seems to work at values around 0.005
model_pipeline = Regressor(module=MyModule, loss_fn='mse', lr=0.01, optimizer_fn='sgd')
trans = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=128, antialias=True)])
metric = metrics.MAE()

input("Waiting to continue...")

while True:
    inp = input("Enter a choice: \n\t1) Try a test prediction \n\t2) Close \n\t3) Save model \n\t4) Load model\n\n> ")

    if inp == '1':
        y = CHEESE
        # put picture of cheese here!
        image = Image.open("data\\cheese.png")
        y_pred = predict_model(model_pipeline, image)
        print("Jimbo predicted:", "%.2f"%(y_pred) + ", the correct answer was:", y)
        print("Jimbo was off by ", "%.2f"%abs(y_pred - y), ".")
        word_true = cob(y)
        word_guess = cob(y_pred)
        outcome = "CORRECT!" if word_guess == word_true else "WRONG!"
        print("Jimbo thinks this is a picture of ", word_guess, ". ", outcome)
        
        y = BREAD
        # put image of bread here!
        image = Image.open("data\\bread.png")
        y_pred = predict_model(model_pipeline, image)
        print("Jimbo predicted:", "%.2f"%(y_pred) + ", the correct answer was:", y)
        print("Jimbo was off by ", "%.2f"%abs(y_pred - y), ".")
        word_true = cob(y)
        word_guess = cob(y_pred)
        outcome = "CORRECT!" if word_guess == word_true else "WRONG!"
        print("Jimbo thinks this is a picture of ", word_guess, ".", outcome)

        
    elif inp == '2':
        print("Closing...")
        exit()

    elif inp == '3':
        print("Saving...")
        with open(PATH, 'wb+') as f:
            pickle.dump(model_pipeline, f)

    elif inp == '4':
        print("Loading...")
        model_pipeline = pickle.load(open(PATH,"rb"))

    elif inp == '5':
        print(f'MAE: {metric.get():.2f}')