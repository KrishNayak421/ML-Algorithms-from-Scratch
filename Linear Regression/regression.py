import matplotlib.pyplot as plt
import pandas as pd
import time

start = time.time()

data = pd.read_csv("train.csv")

def loss_function(w,b,points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].feature
        y = points.iloc[i].target
        total_error += (y - (w*x + b))**2
    total_error = total_error/float(len(points))
    return total_error

def gradient_descent(w_now , b_now, points, L):
    w_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].feature
        y = points.iloc[i].target

        w_gradient += (-2/n)*x*(y - (w_now * x + b_now))
        b_gradient += (-2/n)*(y - (w_now * x + b_now))
    
    w = w_now - w_gradient*L
    b = b_now - b_gradient*L

    return w,b

w = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    w, b = gradient_descent(w,b,data,L)
    if(i%50 == 0):
        print("Epoch: ", i, "Loss: ", loss_function(w,b,data))
print(w,b)

# plt.scatter(data.feature,data.target,color = "red")
# plt.plot(list(range(0,100)), [w*x + b for x in range(0,100)], color = "black")
# plt.show()

### Testing the model
test_data = pd.read_csv("test.csv")
test_data["predictions"] = test_data["feature"].apply(lambda x: w * x + b)
# print(test_data)

# plt.figure(figsize=(6,4))
# plt.scatter(test_data["feature"], test_data["target"], color="blue", label="Actual",s = 20)
# plt.scatter(test_data["feature"], test_data["predictions"], color="red", label="Predicted",s = 20)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.legend()
# plt.title("Actual vs Predicted Values on Test Data")
# plt.show()

end = time.time()
print("Time taken: ", end-start)
