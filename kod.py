# =========================================================
# --- 0. Import bibliotek ---
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import (
    Input, #wejsciowa, wymiary
    Conv2D, #konw. filtry
    BatchNormalization, #stabilizacja, większy lr
    Activation, #relu(nieliniowosc)
    MaxPooling2D, #mniejszy model
    Flatten, #wektor
    Dense, #rozłożenie wiedzy
    Dropout #zerowanie neuronów
)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("exports", exist_ok=True)

# =========================================================
# --- 1. Importowanie danych CIFAR10 ---
# =========================================================
# Wczytanie zbioru danych CIFAR10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train:", x_train.shape)
print("y_train:", y_train.shape)
print("x_test :", x_test.shape)
print("y_test :", y_test.shape)

plt.imshow(x_train[0])
plt.axis('off')
plt.title(f"Etykieta numeryczna: {y_train[0][0]}")
plt.show()

# =========================================================
# --- 2. Wstępne przetwarzanie danych ---
# =========================================================
# Normalizacja do zakresu [0,1] i dodanie kanału (H,W,1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# one-hot encoding etykiet na wektory dla rozroznienia klas
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat  = to_categorical(y_test, num_classes)

# lista nazw klas CIFAR-10 (użyteczna w wyświetleniach)
class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# =========================================================
# --- 3. Definicja modelu sieci neuronowej ---
# =========================================================
input_shape = (32, 32, 3) #obraz wejściowy
weight_decay = 5e-4   #zachowanie małych wag

model = Sequential([
    Input(shape=input_shape, name="input_layer"),
    #1 krawedzie, kolory
    Conv2D(32, (3,3),
           padding='same', #taki sam rozmiar obrazu
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv1_1'),
    BatchNormalization(name='bn1_1'),
    Activation('relu', name='relu1_1'),

    Conv2D(32, (3,3), #zlozone wzorce
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv1_2'),
    BatchNormalization(name='bn1_2'),
    Activation('relu', name='relu1_2'),

    MaxPooling2D((2,2), name='pool1'),
    Dropout(0.25, name='dropout1'),

    #2 czesci obiektow
    Conv2D(64, (3,3),
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv2_1'),
    BatchNormalization(name='bn2_1'),
    Activation('relu', name='relu2_1'),

    Conv2D(64, (3,3),
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv2_2'),
    BatchNormalization(name='bn2_2'),
    Activation('relu', name='relu2_2'),

    MaxPooling2D((2,2), name='pool2'),
    Dropout(0.30, name='dropout2'),

    #3 calosc obrazu
    Conv2D(128, (3,3),
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv3_1'),
    BatchNormalization(name='bn3_1'),
    Activation('relu', name='relu3_1'),

    Conv2D(128, (3,3),
           padding='same',
           kernel_initializer='he_normal',
           kernel_regularizer=l2(weight_decay),
           name='conv3_2'),
    BatchNormalization(name='bn3_2'),
    Activation('relu', name='relu3_2'),

    MaxPooling2D((2,2), name='pool3'),
    Dropout(0.35, name='dropout3'),

    #polaczenie
    Flatten(name='flatten'),

    Dense(256, # 256 neuronow dla CIFAR-10
          activation='relu',
          kernel_initializer='he_normal',
          name='dense1'),
    Dropout(0.4, name='dropout4'),
    #prawdopodobienstwa klas
    Dense(num_classes, activation='softmax', name='output')
])

# =========================================================
# --- 4. Kompilacja modelu ---
# =========================================================
# Optymalizator Adam, funkcja straty categorical_crossentropy, metryka accuracy
optimizer = Adam(learning_rate=7e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy', #roznica wektor softmax a one-hot etykiety
    metrics=['accuracy']
)

# Wyświetlenie podsumowania modelu
model.summary()

# =========================================================
# --- 5. Trenowanie modelu ---
# =========================================================
es = EarlyStopping( #brak poprawy - koniec treningu
    monitor='val_loss',
    patience=20, # liczba epok bez poprawy, po których zatrzymujemy trening
    min_delta=5e-4, # minimalna wymagana zmiana, by uznać, że jest „poprawa”
    mode='min', # 'min' - monitorujemy straty - bledy w prawdopodobienstwach
    restore_best_weights=True
)

rlp = ReduceLROnPlateau( #zmniejszenie lr 2x
    monitor='val_loss',
    factor=0.5, # ile razy zmniejszyć LR (tu: o połowę)
    patience=5, # liczba epok bez poprawy przed zmniejszeniem LR
    min_delta=5e-4,
    min_lr=1e-6, # dolna granica learning rate
    mode='min'
)

history = model.fit( #trenowanie
    x_train,
    y_train_cat,
    batch_size=64,
    epochs=100,
    validation_data=(x_test, y_test_cat),
    callbacks=[es, rlp], #aktywacja es i rlp
    verbose=1
)

# Data skończenia treningu - timestamp - znacznik do zapisywania plików
ts = datetime.datetime.now().strftime("_%Y%m%d_%H%M")

best_val_loss = min(history.history['val_loss'])
best_val_acc  = max(history.history['val_accuracy'])
print(f"Najlepszy val_loss: {best_val_loss:.4f}")
print(f"Najlepsza val_accuracy: {best_val_acc:.4f}")

# =========================================================
# --- 6. Ewaluacja modelu ---
# =========================================================
loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print('Ostateczna dokładność:', f"{accuracy:.4f}")
print('Ostateczna strata (val_loss):', f"{loss:.4f}")

# =========================================================
# --- 7. Wizualizacja przebiegu treningu (loss) ---
# =========================================================
fig_loss_acc = plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss (train)')
plt.plot(history.history['val_loss'], label='loss (val)')
plt.xlabel('Epoka')
plt.ylabel('categorical_crossentropy')
plt.title(f'Najlepszy val_loss = {best_val_loss:.4f}')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy (train)')
plt.plot(history.history['val_accuracy'], label='accuracy (val)')
plt.xlabel('Epoka')
plt.ylabel('Accuracy')
plt.title(f'Najlepsza val_accuracy = {best_val_acc:.4f}')
plt.legend()

plt.tight_layout()
plt.savefig(f"exports/training_loss_accuracy{ts}.png")
plt.show()

# =========================================================
# --- 8. Wizualizacja błędnych klasyfikacji i macierz pomyłek ---
# =========================================================
pred_probs = model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = np.argmax(y_test_cat, axis=1)

# Indeksy błędnych klasyfikacji
incorrect_indices = np.nonzero(pred_labels != true_labels)[0]

# Wyświetlenie kilku błędnych przykładów
n_show = min(6, len(incorrect_indices))
plt.figure(figsize=(12,6))
for i in range(n_show):
    idx = incorrect_indices[i]
    plt.subplot(2, 3, i+1)
    plt.imshow(x_test[idx])
    plt.title(f"GT: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}")
    plt.axis('off')
plt.suptitle("Przykłady błędnych klasyfikacji")
plt.tight_layout()
plt.savefig(f"exports/wrong_examples{ts}.png")
plt.show()

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig_cm, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax, cmap='Blues', colorbar=False, xticks_rotation=45)
plt.title("Macierz pomyłek")
plt.savefig(f"exports/confusion_matrix{ts}.png")
plt.show()
