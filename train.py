import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import load_dataset

# Загрузка датасета
x, y = load_dataset()

# Разделение на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Параметры обучения
epochs = 5
learning_rate = 0.001

# Создание модели
model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

# Получение финальных значений loss и accuracy
final_loss, final_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Training completed")
name = ("trained_model" +
        f"_epochs-{epochs}" +
        f"_rate-{learning_rate}" +
        "_loss-" + str(round(final_loss, 3)) +
        "_acc-" + str(round(final_accuracy, 3)) + ".keras")
print(f"Save model into file \"{name}\"? y/n")

if input().lower() == "y":
    print(f"Saving as {name}")
    model.save(name)
    print("Model saved")
else:
    print("Model not saved")
