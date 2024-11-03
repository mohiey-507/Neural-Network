
import numpy as np
from neural_network import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_iris

def test_binary_classification():

    # Load data
    X, y = load_breast_cancer(return_X_y=True)

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = y.reshape(-1, 1)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)

    # Create and compile model
    model = Network()
    model.add(Dense(32, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile with specified optimizer
    model.compile(loss='binary_cross_entropy',
                optimizer='adam',
                optimizer_params={'learning_rate': 0.0002})
    
    # Train
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=128,
                        validation_data=(X_test, y_test))
    

def test_multi_classification():

    # Load data
    X, y = load_iris(return_X_y=True)

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to one-hot encode labels
    y = np.eye(3)[np.array(y, dtype=int).reshape(-1)]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Create and compile model
    model = Network()
    model.add(Dense(32, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.4))
    model.add(Dense(8, activation='relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    
    # Compile with specified optimizer
    model.compile(loss='categorical_cross_entropy',
                optimizer='adam',
                optimizer_params={'learning_rate': 0.001})
    
    # Train
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_test, y_test))


if __name__ == '__main__':
    test_binary_classification()
    print('='*100)
    test_multi_classification()
