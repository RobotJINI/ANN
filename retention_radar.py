import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
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

def load_train_test(file_name):
    dataset = pd.read_csv(file_name)
    
    #Start with credit score (index 3)
    features = dataset.iloc[:, 3:-1].values
    results = dataset.iloc[:, -1].values
    
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
    features_train, features_test, results_train, results_test = train_test_split(features, results, test_size = 0.2, random_state = 0)
    
    # feature scaling
    sc = StandardScaler()
    features_train = sc.fit_transform(features_train)
    features_test = sc.transform(features_test)
    
    return (features_train, features_test, results_train, results_test)

def create_ann():
    # Initializing the ANN
    ann = tf.keras.models.Sequential()
    
    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    
    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    return ann

def train(ann):
    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Training the ANN on the Training set
    ann.fit(features_train, results_train, batch_size = 32, epochs = 100)


if __name__ == "__main__":
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.debug(tf.__version__)
    
    features_train, features_test, results_train, results_test = load_train_test('Churn_Modelling.csv')
    create_ann()