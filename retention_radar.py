import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import os
import logging

# may put this function in another utility file
def import_tensorflow():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    
    tf.get_logger().setLevel(logging.ERROR)
    
    return tf

def preprocess_data(file_name):
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



if __name__ == "__main__":
    # replace `import tensorflow as tf` with this line
    # or insert this line at the beginning of the `__init__.py` of a package that depends on tensorflow
    tf = import_tensorflow()  
    
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.debug(tf.__version__)
    preprocess_data('Churn_Modelling.csv')