# PhotoLab Express

Esta aplicación te permite aplicar filtros y ejecutar detección de rostros en un lote de imágenes.

## Cómo usar

1. Instala las dependencias: `pip install -r requirements.txt`
2. Ejecuta la aplicación: `streamlit run app.py`
3. Arrastra y suelta una carpeta con imágenes.
4. Selecciona los filtros y otras opciones.
5. Haz clic en "Procesar" para iniciar el procesamiento por lotes.
6. Los resultados se guardarán en las carpetas correspondientes y se podrá descargar un archivo CSV.

## Resultados del Profiling

El profiling se realizó utilizando `cProfile` y `snakeviz`.

```bash
python -m cProfile -o profile.out app.py
```

Los resultados muestran que las operaciones que consumen más tiempo son los filtros de procesamiento de imágenes, especialmente cuando se aplican a imágenes grandes. El uso de una caché mejora significativamente el rendimiento en ejecuciones posteriores con las mismas imágenes y parámetros.
