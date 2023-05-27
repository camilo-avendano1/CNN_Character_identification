from tensorflow.keras.preprocessing import image_dataset_from_directory

# Cargar los datos de entrenamiento y validación con imágenes de tamaño 128x128
train_ds = image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(128, 128),
    batch_size=batch_size)

val_ds = image_dataset_from_directory(
    data_dir,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(128, 128),
    batch_size=batch_size)


from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l1, l2

# Cargar el modelo VGG19 pre-entrenado (sin incluir la capa densa superior)
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Congelar las capas del modelo base para que no se actualicen durante el entrenamiento
base_model.trainable = False

# Agregar una capa densa superior para la clasificación
model = Sequential([
    base_model,
    Flatten(),
    Dense(992, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(248, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(62, activation='softmax')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


epochs=55
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
)
