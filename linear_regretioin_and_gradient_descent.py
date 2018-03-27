import numpy as np
from PIL import Image

flat_arr = None

def loadImage() :
    basewidth = 15 #pixels
    train_x = []
    global flat_arr
    for i in range(12) :
        #open source image
        img = Image.open('./data/' + str(i) + '.png').convert('RGBA')
        #resize image to 25x25
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        imgCom = img.resize((basewidth, hsize), Image.ANTIALIAS)
        #make an image into vector
        arr = np.array(imgCom)
        flat_arr = arr.ravel()
        train_x.append(flat_arr/255.0) #div array became 1 and 0
    return train_x

X = loadImage()
Y = [[0.],[0.],[0.],[0.],[0.],[0.],[1.],[1.],[1.],[1.],[1.],[1.]]
number_of_data = len(flat_arr)

w = np.random.randn(900) #random var m with vector
b = np.random.randn(1) #random var b scalar

def sigmoid(x):
    return 1/(1+np.exp(-x))

def prediction(x):
    res = np.sum(w*x) + b
    # print(np.prod(w*x))
    # print(b)
    return sigmoid(res)

def cost(predict, y):
    res = ((predict - y)**2)/2.0
    # print(res)
    return res

learning_rate = 0.001

total_error = 0.0

for _ in range(5000):
    for x, y in zip(X, Y):
        pred = prediction(x)
        for i in range(len(w)):
            w[i] = w[i] - (learning_rate * (pred[0] - y[0])*x[i])
        b = b - (learning_rate * (pred[0] - y[0]))
        #print(cost(pred[0], y[0]))

for i in X:
    print prediction(i)