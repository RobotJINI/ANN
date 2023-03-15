import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import os
import logging


def import_tensorflow():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    
    tf.get_logger().setLevel(logging.ERROR)
    
    return tf
tf = import_tensorflow()  

class RetentionRadar:
    def __init__(self, file_name):
        self.load_train_test(file_name)
        self.create_ann()
    
    def load_train_test(self, file_name):
        self._dataset = pd.read_csv(file_name)
        
        #Start with credit score (index 3)
        features = self._dataset.iloc[:, 3:-1].values
        results = self._dataset.iloc[:, -1].values
        
        logging.debug(f'features:\n{features}')
        logging.debug(f'dep_variable:\n{results}')
        
        #Label encoding gender column
        le = LabelEncoder()
        features[:, 2] = le.fit_transform(features[:, 2])
        logging.debug(f'features label encoding gender column:\n{features}')
        
        #One hot encoding on Geography column
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
        features = np.array(ct.fit_transform(features))
        logging.debug(f'features one hot encoding on geography column:\n{features}')
        
        # splitting dataset into training and test set
        self._features_train, self._features_test, self._results_train, self._results_test = train_test_split(features, results, test_size = 0.2, random_state = 0)
        
        # feature scaling
        self._sc = StandardScaler()
        self._features_train = self._sc.fit_transform(self._features_train)
        self._features_test = self._sc.transform(self._features_test)
    
    def create_ann(self):
        # Initializing the ANN
        self._ann = tf.keras.models.Sequential()
        
        # Adding the input layer and the first hidden layer
        self._ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        
        # Adding the second hidden layer
        self._ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        
        # Adding the output layer
        self._ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        return self._ann
    
    def train(self):
        # Compiling the ANN
        self._ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        # Training the ANN on the Training set
        self._ann.fit(self._features_train, self._results_train, batch_size = 32, epochs = 100)
        
    def predict_leave_bank(self, new_observations):
        return (self._ann.predict(self._sc.transform(new_observations)) > 0.5)
    
    def test(self):
        pred_test = self._ann.predict(self._features_test)
        pred_test = (pred_test > 0.5)
        
        cm = confusion_matrix(self._results_test, pred_test)
        logging.info(f'confusion matrix:\n{cm}')
        acc_score = accuracy_score(self._results_test, pred_test)
        logging.info(f'accuracy score:\n{acc_score}')


if __name__ == "__main__":
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)
    logging.debug(tf.__version__)
    
    rr = RetentionRadar('Churn_Modelling.csv')
    rr.train()
    
    '''
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40 years old
    Tenure: 3 years
    Balance: $ 60000
    Number of Products: 2
    Does this customer have a credit card? Yes
    Is this customer an Active Member: Yes
    Estimated Salary: $ 50000
    '''
    prediction = rr.predict_leave_bank([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
    logging.info(f'Will customer leave bank? {prediction}')
    
    rr.test()
