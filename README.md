# PhotoLab Express

Esta aplicación te permite aplicar filtros y ejecutar detección de rostros en un lote de imágenes.

## Cómo usar

Sigue estos pasos para ejecutar la aplicación. Se asume que tienes Python y pip instalados.

### 1. Instala las dependencias

Desde la raíz del repositorio, ejecuta el siguiente comando para instalar todas las librerías necesarias:

```bash
pip install -r project/requirements.txt
```

### 2. Ejecuta la aplicación

Una vez instaladas las dependencias, ejecuta la aplicación con este comando, también desde la raíz del repositorio:

```bash
streamlit run project/app.py
```

### 3. Utiliza la aplicación

1.  La aplicación se abrirá en tu navegador.
2.  Arrastra y suelta una o varias imágenes en el área de carga.
3.  Selecciona los filtros que deseas aplicar.
4.  Marca la casilla "Detectar rostros" si lo necesitas.
5.  Haz clic en "Procesar" para iniciar el procesamiento.
6.  Los resultados se mostrarán en pantalla y podrás descargar un archivo CSV con las clasificaciones. Las imágenes procesadas se guardarán en el directorio `output/`.

## Resultados del Profiling

El profiling se realizó utilizando `cProfile` y `snakeviz`.

```bash
python -m cProfile -o profile.out project/app.py
```

Los resultados muestran que las operaciones que consumen más tiempo son los filtros de procesamiento de imágenes, especialmente cuando se aplican a imágenes grandes. El uso de una caché mejora significativamente el rendimiento en ejecuciones posteriores con las mismas imágenes y parámetros.
