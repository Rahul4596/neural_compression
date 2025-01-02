import sys
import numpy as np
from PIL import Image

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from torchsummary import summary

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y, device):
        self.device = device
        self.X = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.y = torch.from_numpy(y.astype(np.float32)).to(self.device)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
    
class PredictNet(nn.Module):
    
    def __init__(self, NFEATURES, NHIDDEN):
        super(PredictNet, self).__init__()
        self.NFEATURES = NFEATURES
        self.fc1 = nn.Linear(NFEATURES, NHIDDEN)
        self.fc2 = nn.Linear(NHIDDEN, 1)

    def forward(self, x):
        # x = x.view(-1, self.NFEATURES)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
class PredictModel():
    
    def __init__(self, NFEATURES, NHIDDEN, device):
        self.model = PredictNet(NFEATURES, NHIDDEN).to(device)
        self.optimiser = Adam(self.model.parameters(), lr=0.002)
        self.device = device
        
    def predict(self, X):
        with torch.no_grad():
            # prepare data
            X = torch.tensor(X, dtype=torch.float).to(self.device)
            # forward pass
            output = self.model(X)
        return output[:,1].numpy()
    
    def fit(self, X, y):
        self.iterate_batch(X, y)
        return
    
def convert_to_bitplanes(image):
    bit_image = np.zeros((image.shape[0] + 2, image.shape[1] + 2, image.shape[2] * 9))
    print(bit_image.shape)

    for i in range(1, image.shape[0] + 1):
        for j in range(1, image.shape[1] + 1):
            for k in range(0, image.shape[2]):
                dec_val = image[i - 1][j - 1][k]
                count = 0
                while(dec_val > 0):
                    bit_image[i][j][k * 9 + (8 - count)] = (dec_val & 1)
                    dec_val = dec_val >> 1
                    count = count + 1
                
                # print(image[i][j][k], bit_image[i][j])

    for i in range(1, bit_image.shape[0] - 1):
        for j in range(1, bit_image.shape[1] - 1):
            for k in range(1, bit_image.shape[2]):
                bit_image[i][j][k] = np.bitwise_xor(np.int_(bit_image[i][j][k]), np.int_(bit_image[i][j][k - 1]))

    return bit_image


def get_input_vector(image, i, j, k):
    input = np.zeros(9)
    
    input[0] = image[i-1][j][k]
    input[1] = image[i][j-1][k]
    input[2] = image[i-1][j-1][k]
    input[3] = image[i-1][j+1][k]

    input[4] = image[i-1][j][k-1]
    input[5] = image[i][j-1][k-1]
    input[6] = image[i-1][j-1][k-1]
    input[7] = image[i-1][j+1][k-1]
    input[8] = image[i][j][k-1]

    # if(k > 9):
    #     input[9] = image[i-1][j][k-9]
    #     input[10] = image[i][j-1][k-9]
    #     input[11] = image[i-1][j-1][k-9]
    #     input[12] = image[i-1][j+1][k-9]
    #     input[13] = image[i][j][k-9]

    return input

def get_full_input_arrays(image, NFEATURES):
    count = 0
    X = np.empty((((image.shape[0] - 2) * (image.shape[1] - 2) * int(image.shape[2] / 9) * 8), NFEATURES))
    Y = np.empty((((image.shape[0] - 2) * (image.shape[1] - 2) * int(image.shape[2] / 9) * 8), 1))
    for k in range(0, int(image.shape[2] / 9)):
        for l in range(8):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    input_vector = get_input_vector(image, i, j, k * 9 + l + 1)

                    for m in range(NFEATURES):
                        X[count][m] = input_vector[m]
                    
                    Y[count][0] = image[i][j][k * 9 + l + 1]
                    
                    count += 1

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def train_network(image, NFEATURES, batch_size, num_epochs):
    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    print(f"Using {device} device")

    X, Y = get_full_input_arrays(image, NFEATURES)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                   random_state=104,  
                                   test_size=0.1,  
                                   shuffle=True) 

    print(X.shape, Y.shape)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # Instantiate training and test data
    train_data = Data(X_train, y_train, device)
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    test_data = Data(X_test, y_test,device)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Check it's working
    num_epochs = 10
    loss_values = []

    predict_model = PredictModel(NFEATURES, 20, device)

    summary(predict_model.model, (64, 9))

    for epoch in range(num_epochs):
        for X_1, y_1 in train_dataloader:
            # zero the parameter gradients
            predict_model.optimiser.zero_grad()
       
            # forward + backward + optimize
            pred = predict_model.model(X_1)
            loss = F.mse_loss(pred, y_1.unsqueeze(-1))
            print("loss = ", loss.item())
            loss.backward()
            predict_model.optimiser.step()
        
        print("epoch ", epoch, " done")

         # Initialize required variables
        y_pred = []
        y_test = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_1, y_1 in test_dataloader:
                outputs = predict_model.model(X_1)  # Get model outputs
                predicted = outputs.cpu().numpy()  # Convert to NumPy and apply threshold
                # predicted = list(itertools.chain(*predicted))  # Flatten predictions
                # y_pred.append(predicted)  # Append predictions
                # y_test.append(y_1.cpu().numpy())  # Append true labels as NumPy
                total += predicted.shape[0]  # Increment total count
                correct += np.power(predicted - y_1.cpu().numpy(), 2).sum()  # Count correct predictions
                print(np.transpose(predicted), np.transpose(y_1.cpu().numpy()))

        print(f'Accuracy of the test instances: {100 * correct // total}%')

   

    

# loading the image
png_pil_img = Image.open(sys.argv[1])
png_np_img = np.asarray(png_pil_img)

if(png_np_img.ndim == 2):
    png_np_img = png_np_img.reshape((png_np_img.shape[0], png_np_img.shape[1], 1))

bit_image = convert_to_bitplanes(png_np_img)
train_network(bit_image, 9, 32, 10)

