import numpy as np

X = np.array([[1,2]])
W1 = np.array([[1,1],[1,0]])
W2 = np.array([[0,1],[1,1]])
b1 = np.zeros(2)
b2 = np.zeros(2)

def softmax(a):
    expA = np.exp(a)
    return expA/expA.sum(axis=1,keepdims=True)

def forward(X,W1,b1,W2,b2):
    Z=np.tanh(X.dot(W1)+b1)
    return softmax(Z.dot(W2)+b2)

P_Y_given_X = forward(X,W1,b1,W2,b2)
predictions = np.argmax(P_Y_given_X,axis=1)

print(P_Y_given_X)