#include <Arduino.h>
#include <math.h>

// Definicion de pines de hardware
const int PIN_POT   = 20;  // Entrada ADC: Potenciometro de 50 kOhm
const int PIN_TRIG  = 5;   // Salida digital: Trigger HC-SR04
const int PIN_ECHO  = 4;   // Entrada digital: Echo HC-SR04
const int PIN_SERVO = 32;  // Salida PWM: Servomotor MG90S

// Parametros de configuracion PWM (50 Hz, 12 bits)
const int PWM_FREQ = 50;
const int PWM_RES  = 12;

// Variables de estado para filtrado exponencial
float dist_filtrada_cm  = 50.0f;
float angulo_actual     = 0.0f;
int   lecturas_perdidas = 0;

// Matrices de pesos y vectores de sesgo 
const float W1[2][4] = {
  {  1.4285, -0.8512,  1.1940,  0.9124 },
  { -1.6520, -1.1045, -1.3890, -0.9540 }
};
const float b1[4] = { -0.1240, 0.9850, -0.0520, 0.1120 };

const float W2[4][1] = {
  {  1.8920 },
  { -2.1450 },
  {  1.7640 },
  {  1.2310 }
};
const float b2[1] = { -1.4210 };

// Funciones de activacion
float relu(float x) {
  return (x > 0.0f) ? x : 0.0f;
}

float sigmoid(float x) {
  if (x < -15.0f) return 0.0f;
  if (x >  15.0f) return 1.0f;
  return 1.0f / (1.0f + expf(-x));
}

// Propagacion hacia adelante 
float predecirRedNeuronal(float x1_vel, float x2_dist) {
  float ocultas[4];
  for (int j = 0; j < 4; j++) {
    float suma = (x1_vel * W1[0][j]) + (x2_dist * W1[1][j]) + b1[j];
    ocultas[j] = relu(suma);
  }

  float suma_salida = b2[0];
  for (int j = 0; j < 4; j++) {
    suma_salida += ocultas[j] * W2[j][0];
  }

  return sigmoid(suma_salida);
}

// Lectura y normalizacion del potenciometro (0.0 a 1.0)
float leerPotenciometroNorm() {
  long suma = 0;
  for (int i = 0; i < 8; i++) {
    suma += analogRead(PIN_POT);
    delay(1);
  }
  float promedio = suma / 8.0f;
  float norm = promedio / 4095.0f;
  return constrain(norm, 0.0f, 1.0f);
}

// Medicion y filtrado de distancia con sensor HC-SR04 (5 cm a 50 cm)
float leerDistanciaEstableCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duracion = pulseIn(PIN_ECHO, HIGH, 25000);
  float nueva_dist;

  if (duracion == 0) {
    lecturas_perdidas++;
    if (lecturas_perdidas > 8) {
      nueva_dist = 50.0f;
    } else {
      nueva_dist = dist_filtrada_cm;
    }
  } else {
    float cm = (duracion * 0.0343f) / 2.0f;
    if (cm < 2.0f) {
      nueva_dist = dist_filtrada_cm;
    } else {
      lecturas_perdidas = 0;
      nueva_dist = constrain(cm, 5.0f, 50.0f);
    }
  }

  // Filtro de media movil exponencial (EMA)
  dist_filtrada_cm = (0.70f * dist_filtrada_cm) + (0.30f * nueva_dist);
  return dist_filtrada_cm;
}

// Control de posicion del servomotor MG90S
void moverServo(float angulo) {
  angulo = constrain(angulo, 10.0f, 170.0f);
  int duty = (int)(102.0f + (angulo / 180.0f) * (491.0f - 102.0f));
  ledcWrite(PIN_SERVO, duty);
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  analogReadResolution(12);

  ledcAttach(PIN_SERVO, PWM_FREQ, PWM_RES);
  moverServo(10.0f);
}

void loop() {
  // Adquisicion y normalizacion de entradas
  float x1_vel  = leerPotenciometroNorm();
  float dist_cm = leerDistanciaEstableCm();
  float x2_dist = (dist_cm - 5.0f) / (50.0f - 5.0f);

  // Inferencia de la red neuronal y calculo de referencia
  float y_nn = predecirRedNeuronal(x1_vel, x2_dist);
  float y_esperado = constrain(x1_vel * powf(1.0f - x2_dist, 1.5f), 0.0f, 1.0f);

  // Actualizacion suavizada del actuador
  float angulo_objetivo = y_nn * 180.0f;
  angulo_actual = (0.75f * angulo_actual) + (0.25f * angulo_objetivo);
  moverServo(angulo_actual);

  // Monitoreo serie para adquisicion de graficas
  static int contador_print = 0;
  contador_print++;
  if (contador_print >= 2) {
    contador_print = 0;
    Serial.print("Vel_Norm:");
    Serial.print(x1_vel, 2);
    Serial.print("\tDist_Norm:");
    Serial.print(x2_dist, 2);
    Serial.print("\tSalida_Red:");
    Serial.print(y_nn, 2);
    Serial.print("\tEsperado:");
    Serial.print(y_esperado, 2);
    Serial.print("\tAngulo:");
    Serial.println(angulo_actual, 1);
  }

  delay(40);
}
