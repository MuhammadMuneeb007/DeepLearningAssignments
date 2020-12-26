import matplotlib.pyplot as plt 
import numpy as np 
import math 


def sigmoid(number):
    return 1/(1 + np.exp(-number))
def dsigmoid(number):
    f = 1/(1 + np.exp(-number))
    return f * (1 - f)

number=10
vector =  np.array([12,4,2,0])
matrix=  np.array([[5, 78, 2, 34, 0],
 [6, 79, 3, 35, 1],
 [7, 80, 4, 36, 2]])


print("Number",number)
print("Sigmoid",sigmoid(number))
print("Derivative",dsigmoid(number))

print("Vector",vector)
print("Sigmoid",sigmoid(vector))
print("Derivative",dsigmoid(vector))

print("Matrix",matrix)
print("Sigmoid",sigmoid(matrix))
print("Derivative",dsigmoid(matrix))



