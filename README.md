# RetentionRadar
RetentionRadar is a Python class that trains an Artificial Neural Network (ANN) model to predict whether a bank customer will leave the bank or not. The model is trained on the "Churn_Modelling.csv" dataset and uses features such as the credit score, gender, age, balance, and estimated salary of the customer to make predictions.

## Requirements
* pandas
* scikit-learn
* tensorflow

## Installation
You can install the required dependencies using pip:

```bash
pip install pandas scikit-learn tensorflow
```

## Usage
To use RetentionRadar, you can create an instance of the class and pass in the filename of the dataset:

```python
from RetentionRadar import RetentionRadar

rr = RetentionRadar('Churn_Modelling.csv')
```

The RetentionRadar object will automatically load and preprocess the dataset, create an ANN model, and train it on the training data.

To make predictions for new customers, you can call the predict_leave_bank() method of the RetentionRadar object and pass in a 2D array of new observations:

```python
prediction = rr.predict_leave_bank([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
```

This will return a boolean value indicating whether the customer is likely to leave the bank or not.

To evaluate the performance of the model, you can call the test() method of the RetentionRadar object:

```python
rr.test()
```

This will print the confusion matrix and accuracy score of the model on the test data.