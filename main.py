from sklearn import svm 
import pandas as pd
from sklearn.preprocessing import LabelEncoder


print("Welcome to the Iris Flower Prediction Program! \n")
print("We will use Support Vector Machines to predict what kind of iris flower you have! \n")
print("All you need to do is supply some information about the flower! \n") #

while True:
  print("1. Make a prediction")
  print("2. Exit the program")

  choice = input("")
  if choice == "1":
    sepalLength = float(input("What is the Sepal Length in cm? \n"))
    sepalWidth = float(input("What is the Sepal Width in cm? \n"))
    petalLength = float(input("What is the Petal Length in cm? \n"))
    petalWidth = float(input("What is Petal Width in cm? \n"))

    df = pd.read_csv("Iris.csv")
    X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    Y = df[['Species']]

    le = LabelEncoder()
    yEncoded = le.fit_transform(Y['Species'])

    irisPredictionModel = svm.SVC()
    irisPredictionModel.fit(X, yEncoded)

    prediction = irisPredictionModel.predict([ [ sepalLength,  sepalWidth ,  petalLength, petalWidth]] )

    returnToOriginal = le.inverse_transform(prediction)
    print("The type of iris flower is " + returnToOriginal[0])
  elif choice == "2":
    break

print("Goodbye!")