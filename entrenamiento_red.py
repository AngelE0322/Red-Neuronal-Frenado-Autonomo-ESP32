import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Generacion de la base de datos (120 muestras)
n_muestras = 120
sensor_velocidad = np.round(np.random.uniform(0.0, 1.0, n_muestras), 4)
sensor_distancia = np.round(np.random.uniform(0.0, 1.0, n_muestras), 4)

# Funcion objetivo no lineal para el control del actuador
actuador_servo = np.round(
    np.clip(sensor_velocidad * ((1.0 - sensor_distancia) ** 1.5), 0.0, 1.0), 4
)

# Exportacion a archivo CSV
df = pd.DataFrame({
    'sensor_velocidad': sensor_velocidad,
    'sensor_distancia': sensor_distancia,
    'actuador_servo': actuador_servo
})
df.to_csv('base_datos_sensores.csv', index=False)

# Separacion de variables de entrada (X) y salida (y)
X = df[['sensor_velocidad', 'sensor_distancia']].values
y = df['actuador_servo'].values

# Arquitectura de la red neuronal (MLP: 2-4-1)
model = Sequential([
    Dense(4, input_dim=2, activation='relu', name='Capa_Oculta'),
    Dense(1, activation='sigmoid', name='Capa_Salida')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
    loss='mse',
    metrics=['mae']
)

# Entrenamiento del modelo
historial = model.fit(
    X, y,
    epochs=400,
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

print(f"MSE final: {historial.history['loss'][-1]:.6f}")
print(f"MAE final: {historial.history['mae'][-1]:.4f}")

# Graficas de evaluacion del entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(historial.history['loss'], label='Entrenamiento (MSE)', color='blue')
plt.plot(historial.history['val_loss'], label='Validacion (MSE)', color='orange', linestyle='--')
plt.title('Evolucion del Error Cuadratico Medio (MSE)')
plt.xlabel('Epoca')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
predicciones = model.predict(X, verbose=0).flatten()
plt.scatter(y, predicciones, alpha=0.7, color='green', label='Datos de prueba')
plt.plot([0, 1], [0, 1], 'r--', label='Referencia ideal')
plt.title('Respuesta Esperada vs. Salida de la Red')
plt.xlabel('Valor Esperado')
plt.ylabel('Valor Predicho')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('grafica_entrenamiento.png', dpi=300)
plt.show()

# Extraccion de pesos y sesgos entrenados
W1, b1 = model.layers[0].get_weights()
W2, b2 = model.layers[1].get_weights()

print("W1 =", np.round(W1, 4).tolist())
print("b1 =", np.round(b1, 4).tolist())
print("W2 =", np.round(W2, 4).tolist())
print("b2 =", np.round(b2, 4).tolist())