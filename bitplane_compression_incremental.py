import sys
import arithmeticcoding

from PIL import Image
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def sigmoid_derivative(z):
    return (sigmoid(z) * (1 - sigmoid(z)))

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0

def Leaky_ReLU(Z):
    return np.maximum(0.05 * Z, Z)

def Leaky_ReLU_deriv(Z):
    return np.maximum(Z > 0, 1/20)

class PredictNet:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        dz2 = 2 * (y_pred - y_true)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * ReLU_deriv(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000):

        for epoch in range(epochs):
            y_pred = self.forward(X)
            
            # loss = (y - y_pred) * (y - y_pred)
            
            self.backward(X, y, y_pred)

            # print(self.W1.T)
            # print(self.b1)
            # print(self.W2.T)
            # print(self.b2)


        # if epoch % 100 == 0:
        #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return y_pred


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

    # else:
    #     input[9] = 0
    #     input[10] = 0
    #     input[11] = 0
    #     input[12] = 0
    #     input[13] = 0

    # if(i > 1 and j > 1):
    #     input[14] = image[i-2][j][k]
    #     input[15] = image[i][j-2][k]
    #     input[16] = image[i-2][j-2][k]
    #     input[17] = image[i-2][j+1][k]
    #     input[18] = image[i-2][j-1][k]
    #     input[19] = image[i-2][j+1][k]
    #     input[20] = image[i-1][j-2][k]
    #     input[21] = image[i+1][j-2][k]

    # else:
    #     input[14] = 0
    #     input[15] = 0
    #     input[16] = 0
    #     input[17] = 0
    #     input[18] = 0
    #     input[19] = 0
    #     input[20] = 0
    #     input[21] = 0

    v_norm = input / (np.linalg.norm(input) + 1e-16)

    return v_norm

def compress_image(image, predict_model):
    count = 0
    sum = 0
    iter = 0
    loss = 0
    
    for k in range(0, int(image.shape[2] / 9)):
        for l in range(8):
            for i in range(1, image.shape[0] - 1):
                for j in range(1, image.shape[1] - 1):
                    input_vector = get_input_vector(image, i, j, k * 9 + l + 1)
                    X = np.array(input_vector)
                    Y = np.array(image[i][j][k * 9 + l + 1])
                    # print(X.size)
                    X = X.reshape(1, X.shape[0])

                    pred = predict_model.predict(X)
                    # if(count % 2 == 0):
                    predict_model.train(X, Y, 5)
                                       
                    count = count + 1
                    sum = sum + (pred - Y) * (pred - Y)

                    if(count % 1000 == 0):
                        print(i, j, Y, pred)
                        if(sum / count > loss):
                            iter += 1
                        
                        # if(iter == 5):
                        #     print(predict_model.W1, predict_model.b1, predict_model.W2, predict_model.b2)
                        #     predict_model.W1 = predict_model.W1 + np.random.randn(22, 15) * 0.01
                        #     predict_model.b1 = predict_model.b1 + np.random.randn(1, 15) * 0.01
                        #     predict_model.W2 = predict_model.W2 + np.random.randn(15, 1) * 0.01
                        #     predict_model.b2 = predict_model.b2 * + np.random.randn(1, 1) * 0.01
                        #     iter = 0

                            
                        
                        loss = sum / count

                        print("loss:", loss)

                    # if(count % 10000 == 0):
                    #     predict_model.learning_rate = predict_model.learning_rate * (9 / 10)


   


predict_model = PredictNet(9, 10, 1, 0.005)

# loading the image
png_pil_img = Image.open(sys.argv[1])
png_np_img = np.asarray(png_pil_img)

if(png_np_img.ndim == 2):
    png_np_img = png_np_img.reshape((png_np_img.shape[0], png_np_img.shape[1], 1))

bit_image = convert_to_bitplanes(png_np_img)
compress_image(bit_image, predict_model)

