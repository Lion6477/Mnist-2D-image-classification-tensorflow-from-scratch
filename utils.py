import numpy as np

def load_dataset():
    with np.load("mnist.npz") as f:
        # Конвертация изображений из RGB в единичный RGB
        x_train = f['x_train'].astype(np.float32) / 255
        x_train = x_train.reshape(x_train.shape[0], -1)

        # Метки
        y_train = f['y_train']
        return x_train, y_train
