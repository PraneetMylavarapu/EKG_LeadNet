import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plot_learning(history, title):
    fig, axs = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout(pad=5.0)


    axs[0].plot(history.history['loss'], label='loss')
    axs[1].plot(history.history['acc'], label='accuracy')
    axs[0].plot(history.history['val_loss'], label='val loss')
    axs[1].plot(history.history['val_acc'], label='val accuracy')

    axs[0].legend()
    axs[1].legend()
    fig.savefig('./graphs/' + title + '.png')

def evaluate_model(model, X_test, y_test):
    metrics = model.evaluate(X_test, y_test)
    print('\nEvaluating...')
    print('acc:', metrics[1])
    print('loss:', metrics[0])
    print()

    predictions = model.predict(X_test)

    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test, predictions):
        cnf_matrix[y, round(y_hat[0])] += 1
    print(cnf_matrix)

    return {'loss': metrics[0], 'acc': metrics[1]}

def baseline_feature_network(ekgs: np.ndarray, target: str, lr=0.0001):

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(ekgs, target, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss = tf.keras.losses.binary_crossentropy,
        metrics = ['acc'],
    )

    history = model.fit(
        X_train, 
        y_train, 
        epochs=500,
        validation_data=(X_valid, y_valid),
    )

    evaluate_model(model, X_test, y_test)
    plot_learning(history, 'baseline')

def cnn(data: pd.DataFrame, ekgs: np.ndarray, target: str, lr=1e-6):
    # Get labels
    y = data[target]

    # Reshape the data
    input_data = np.expand_dims(ekgs, axis=-1)

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    
    r_method = 'l2'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=16, kernel_size=10, activation='relu', kernel_regularizer=r_method, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.MaxPooling1D(pool_size=10, padding='same'),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, activation='relu', padding='same', kernel_regularizer=r_method),
        tf.keras.layers.MaxPooling1D(pool_size=10, padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', kernel_regularizer=r_method),
        tf.keras.layers.MaxPooling1D(pool_size=10, padding='same'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=r_method),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=r_method),
        tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=r_method),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
        loss = 'BinaryCrossentropy',
        metrics = ['acc'],
    )

    history = model.fit(
    X_train, 
    y_train, 
    epochs=1500,
    validation_data=(X_valid, y_valid),
    )

    results = evaluate_model(model, X_test, y_test)
    plot_learning(history, 'cnn')

    return results
