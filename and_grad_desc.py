import numpy as np

x_train = np.array([[0.,0.], [0.,1.], [1.,0.], [1.,1.]])
y_train = np.array([[0.], [0.], [0.], [1.]])

w = np.random.randn(2)
b = np.random.randn(1)

num_rate = len(x_train)

def prediction(x):
    return np.sum(w * x) + b

def cost(predict, y):
    res = (predict - y)**2/2
    # print res
    return res

learn_rate = 0.01
error_rate = 0.

print "Prediksi awal "

print "0 0 = ",prediction(np.array([0.,0.]))
print "0 1 = ",prediction(np.array([0.,1.]))
print "1 0 = ",prediction(np.array([1.,0.]))
print "1 1 = ",prediction(np.array([1.,1.]))

# training process
for _ in range(10000) :
    for x, y in zip(x_train, y_train):
        pred = prediction(x)
        for k in range(len(w)):
            w[k] = w[k] - (learn_rate * (pred[0] - y[0]) * x[k])
        b = b - (learn_rate * (pred[0] - y[0]))
    #print(cost(prediction(x[0]), y[0]))

print "\nPrediksi setelah training "

print "0 0 = ",prediction(np.array([0.,0.]))
print "0 1 = ",prediction(np.array([0.,1.]))
print "1 0 = ",prediction(np.array([1.,0.]))
print "1 1 = ",prediction(np.array([1.,1.]))
