# Eye Sentinel

MVP en Python que usa la webcam para medir la apertura relativa de ambos ojos, diferencia
parpadeos naturales de cierres prolongados y activa una alarma local. El vídeo no se guarda
ni se envía a ningún servidor.

> **Aviso:** es un proyecto experimental, no un dispositivo médico ni un sistema de seguridad
> certificado. No debe utilizarse como única protección al conducir o manejar maquinaria.

## Qué incluye el MVP

- Una cara y una webcam local.
- MediaPipe Face Landmarker para localizar seis puntos de cada ojo.
- EAR (Eye Aspect Ratio) filtrado con una mediana corta.
- Calibración personal: frontal, abajo/arriba, izquierda/derecha, parpadeos y cierre.
- Apertura relativa independiente para cada ojo.
- Estimación de orientación `pitch`, `yaw` y `roll`, calibrada respecto a la postura neutral.
- Corrección del EAR por los ángulos vertical y horizontal aprendidos para cada persona.
- Máquina temporal con histéresis: despierto, parpadeo, cierre, alerta y tracking perdido.
- Alarma repetida cuando la apertura relativa permanece por debajo de `0.75` durante 1 s.
- Vista de depuración con landmarks, EAR, apertura, duración, FPS, contador y gráfica.
- Perfil de calibración persistente en `data/calibration.json`.
- Pruebas unitarias de la geometría, calibración y lógica temporal.

PERCLOS, dirección de mirada, varias caras y reconocimiento de identidad quedan
deliberadamente fuera de este MVP.

## Requisitos

- Python 3.11 o 3.12. MediaPipe todavía no publica wheel para Python 3.14.
- macOS, Linux o Windows con una webcam accesible.
- Acceso a Internet la primera vez para descargar el modelo oficial de MediaPipe (~30 MB).

En este equipo ya existe `uv` y Python 3.11, por lo que la ruta recomendada es:

```bash
uv sync --extra dev
uv run eye-sentinel
```

La primera ejecución descargará `models/face_landmarker.task`. macOS puede pedir permiso de
cámara para Terminal/T3 Code; hay que aceptarlo.

Si macOS la bloqueó previamente, actívala en **Ajustes del Sistema → Privacidad y seguridad →
Cámara** para la aplicación desde la que ejecutas el comando y vuelve a abrirla.

## Uso

```bash
# Arranque normal; reutiliza una calibración guardada
uv run eye-sentinel

# Forzar una calibración nueva
uv run eye-sentinel --recalibrate

# Elegir otra cámara
uv run eye-sentinel --camera 1

# Usar otro archivo de configuración
uv run eye-sentinel --config config/default.toml
```

Controles dentro de la ventana:

- `Q` o `Esc`: salir.
- `R`: recalibrar.
- `M`: silenciar o reactivar la alarma durante la sesión.

Durante la calibración sigue las instrucciones para mirar de frente, abajo, arriba, a la izquierda
y a la derecha sin cerrar ni entornar los ojos. Mantén una distancia estable y usa luz uniforme.
Si las posturas o las referencias abierta/cerrada no se distinguen suficientemente, el proceso
se rechaza en lugar de guardar un perfil malo.

## Configuración

Todos los umbrales están en [`config/default.toml`](config/default.toml). Los más relevantes:

```toml
[detector]
closed_threshold = 0.75
reopened_threshold = 0.85
minimum_blink_seconds = 0.08
maximum_blink_seconds = 0.40
alert_after_closed_seconds = 1.00
```

Los umbrales `closed_threshold` y `reopened_threshold` se aplican a la apertura relativa ya
calibrada, no al EAR bruto. La diferencia entre ambos crea histéresis y evita oscilaciones.

La calibración aprende los extremos verticales y horizontales. La apertura esperada de cada ojo
se interpola por separado, ya que al girar la cabeza un ojo puede deformarse más que el otro:

```toml
[head_pose]
pitch_range_margin = 5.0
yaw_range_margin = 5.0
maximum_roll_delta = 20.0
minimum_calibration_pitch_span = 12.0
minimum_calibration_yaw_span = 15.0
```

La aplicación acepta los intervalos que tú hayas recorrido, más cinco grados de margen en cada
eje. Solo fuera de esos intervalos muestra `ÁNGULO TODAVÍA NO CALIBRADO`, reinicia cualquier
cierre en curso y evita la alarma. Es necesario recalibrar tras actualizar desde una versión
anterior.

## Desarrollo y pruebas

```bash
uv run pytest
uv run ruff check .
```

La lógica central vive en `src/eyetracker/services/` y no importa OpenCV ni MediaPipe. Los
adaptadores de cámara, tracking, sonido, pantalla y persistencia están en
`src/eyetracker/adapters/`.

## Estructura

```text
src/eyetracker/
├── application.py          # Orquestación de calibración y monitorización
├── bootstrap.py            # Conecta las implementaciones
├── config.py               # Configuración tipada
├── domain.py               # Modelos y estados comunes
├── ports.py                # Contratos de infraestructura
├── adapters/               # OpenCV, MediaPipe, sonido y JSON
└── services/               # EAR, calibración y detector temporal
```
