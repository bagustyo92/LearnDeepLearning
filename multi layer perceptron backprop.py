import numpy as np
from PIL import Image

flat_arr = None
learning_rate = 0.03
total_error = 0.0

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
number_of_layer = 2
w = np.random.randn(900) #random var m with vector
b = np.random.randn(1) #random var b scalar

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derSigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def prediction(x):
    # print("b : ", b)
    res = np.sum(w*x) + b
    return sigmoid(res)

def cost(predict, y):
    res = ((predict - y)**2)/2.0
    # print(res)
    return res

def backprop(a, y):
    do_b = [np.zeros(bias.shape) for bias in b]
    do_w = [np.zeros(weight.shape) for weight in w]
    activation = a
    # print(a)
    activations = [a]
    # print activations
    zs = []

    for bias, weight in zip(b, w):
        z = np.dot(weight, activation)+b
        zs.append(z)
        activation = sigmoid(z)
        activations.append(activation)

    delta = (activations[-1] - y) * derSigmoid(zs[-1])
    do_b[-1] = delta
    do_w[-1] = np.dot(delta, activations[-2])
    # print(do_w)

    for l in xrange(2, number_of_layer):
        z = zs[-l]
        sp = derSigmoid(z)
        delta = np.dot(weight[-l+1].transpose(), delta) * sp
        do_b [-l] = delta
        do_w [-l] = np.dot(delta, activations[-l-1].transpose())
    return do_b, do_w, z

def output_error(z) :
    return  np.dot(cost(prediction(X[0]), Y[0]), derSigmoid(z))


# B, W = backprop(X[0], Y)
# print(len(X[0])) #900
# print(len(B[0][11]), len(W)) #900, 900
count = 1
for _ in range(100):
    for i in range(len(X)):
        for j in range(len(w)) :
            B, W, Z = backprop(X[i][j], Y[i])
            w[j] = w[j] - learning_rate * W[j]
    b = b - learning_rate * B[0]
    total_error = output_error(Z)
    print "Error ", count, " : ", total_error
    count += 1

for i in X:
    print sigmoid(prediction(i))

#for _ in range(1000):
# for x, y in zip(X, Y):
#     b_prop, w_prop = backprop(x, y)
#     for i in range(len(w)):
#         for j in xrange(len(w_prop)):
#             w[i] = w[i] - learning_rate * w_prop[j]
#             print(w[i])
#     for k in xrange(len(b_prop)):
#         b = b - learning_rate * b_prop[k]
# b_prop, w_prop = backprop(X[0], Y)
# print(len(b_prop[0][0]))
# print(w_prop)

# print(b)

# for _ in range(10000):
#     for x, y in zip(X, Y):
#         pred = sigmoid(prediction(x))
#         for i in range(len(w)):
#             w[i] = w[i] - (learning_rate * (pred[0] - y[0])*x[i])
#         b = b - (learning_rate * (pred[0] - y[0]))
#
# for _ in range(1000):
#     for i in range(len(w)):
#         for j in range(len(w_prop)):
            # print(w_prop[j])
    # for k in range(len(b_prop)):
    #     b = b - learning_rate * b_prop[k]