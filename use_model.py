import numpy as np
import tensorflow as tf
from utils import load_dataset

# Загрузка датасета
x, y = load_dataset()

# Загрузка модели
model_path = input("Enter the path of the saved model: ")
model = tf.keras.models.load_model(model_path)

# Пример использования модели для предсказаний
sample_index = np.random.randint(0, x.shape[0])
sample = x[sample_index].reshape(1, -1)
prediction = model.predict(sample)
predicted_class = np.argmax(prediction)

print(f"Predicted class for sample {sample_index}: {predicted_class}")
print(f"Actual class for sample {sample_index}: {y[sample_index]}")
