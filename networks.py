import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def baseline_network(data: pd.DataFrame, ekgs: np.ndarray, target: str):
    feature_labels = data.columns.copy().tolist()
    feature_labels.remove(target)

    X = np.zeros((data.shape[0], data.shape[1]-1 + ekgs.shape[-1]))
    X[:, :data.shape[1]-1] = data[feature_labels].to_numpy()
    X[:, -ekgs.shape[-1]:] = ekgs[:, 1, :]
    y = data[target]

    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=.0001),
        loss = 'BinaryCrossentropy',
        metrics = ['acc'],
    )

    history = model.fit(
    X_train, 
    y_train, 
    epochs=100,
    validation_data=(X_valid, y_valid),
    )

    metrics = model.evaluate(X_test, y_test)
    print('\nEvaluating...')
    print('acc:', metrics[1])
    print('loss:', metrics[0])
    print()

    predictions = model.predict(X_test)

    cnf_matrix = np.zeros((2, 2))
    for y, y_hat in zip(y_test, predictions):
        cnf_matrix[y, int(y_hat)] += 1
    print(cnf_matrix)

    fig, axs = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    fig.tight_layout(pad=5.0)


    axs[0].plot(history.history['loss'], label='loss')
    axs[1].plot(history.history['acc'], label='accuracy')
    axs[0].plot(history.history['val_loss'], label='val loss')
    axs[1].plot(history.history['val_acc'], label='val accuracy')

    axs[0].legend()
    axs[1].legend()
    fig.savefig('garbage.png')