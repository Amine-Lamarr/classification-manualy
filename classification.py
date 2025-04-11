import numpy as np
import pandas as pd

data = {
    "Hours_Studied": [1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10],
    "Passed": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 
}
data = pd.DataFrame(data)
x  = data['Hours_Studied']
y = data['Passed'].astype(float)
y = y.to_numpy()
x = np.c_[np.ones(x.shape[0]) , x ]
w = np.random.rand(2)

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost function
def CostFunction(x , y , w):
    n = len(y)
    predict = sigmoid(np.dot(x ,w))
    first = np.multiply(-y , np.log(predict))
    second = np.multiply((1 -  y) , np.log(1 - predict))
    return np.sum(first - second) /  n
loss =  CostFunction(x ,y ,w )

# Gradient descent 
def GradientDescnet(x ,y , w, alpha , iters):
    cost =  np.zeros(iters)
    m = len(y)
    for i in range(iters):
        predict = sigmoid(np.dot(x , w))
        dw = (1 / m) * np.dot(x.T, (predict - y))
        w = w -  alpha * dw
        cost[i] =  CostFunction(x ,y ,w)
    return cost , w


alpha = 0.1
iters = 3000
new_cost , new_w = GradientDescnet(x ,y ,w ,alpha , iters)
# results 
print(f"cost before : {loss:.02f}")
print(f'cost after : {new_cost[-1]:.02f}')
print(f'weights before : {w}')
print(f'weights after : {new_w}')

# predictions
def predict(x, w):
    predictions = sigmoid(np.dot(x, w))
    return [1 if i >= 0.5 else 0 for i in predictions]


predictions = predict(x, new_w)

# accuracy
def accuracy(y, predictions):
    correct = np.sum(y == predictions)
    return correct / len(y) * 100

acc = accuracy(y, predictions)
print(f"Accuracy: {acc:.2f}%")

import matplotlib.pyplot as plt

# plotting 
plt.figure(figsize=(10, 6))
plt.plot(range(len(new_cost)), new_cost, color='blue', linewidth=2)
plt.title('Loss Over Iterations', fontsize=16)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss (Cost)', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


