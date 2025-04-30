let modelo;

//Entrenar el modelo al cargar el archivo
window.onload = () => {
  entrenarModelo();
};

// Creación y entrenamiento del modelo
async function entrenarModelo() {
  // Creamos el modelo secuencial
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  modelo.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

  // Datos de entrenamiento
  const xTrain = tf.tensor2d([-6, -5, -4, -3, -2, -1, 0, 1, 2], [9, 1]);
  const yTrain = tf.tensor2d([-6, -4, -2, 0, 2, 4, 6, 8, 10], [9, 1]);

  // Entrenamiento el modelo
  await modelo.fit(xTrain, yTrain, {
    epochs: 350, //cantidad de epocas que se nos solicitó.
    callbacks: {
      onTrainEnd: () => {
        document.getElementById("estado").innerText = " El Modelo está entrenado y listo para usarse.";
        document.getElementById("prediccion").style.display = "block";
      }
    }
  });
}

// Predecir valor de Y a partir del input X
function predecir() {
  const x = parseFloat(document.getElementById("inputX").value);
  if (isNaN(x)) {
    document.getElementById("resultado").innerText = "Ingrese un número válido.";
    return;
  }

  const resultado = modelo.predict(tf.tensor2d([x], [1, 1]));
  resultado.array().then(data => {
    document.getElementById("resultado").innerText = `Predicción para Y = ${data[0][0].toFixed(2)}`;
  });
}
