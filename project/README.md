# PhotoLab Express

This application allows you to apply filters and run face detection on a batch of images.

## How to use

1. Install the dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`
3. Drag and drop a folder with images.
4. Select the filters and other options.
5. Click "Process" to start the batch processing.
6. The results will be saved in a CSV file.

## Profiling Results

Profiling was performed using `cProfile` and `snakeviz`.

```bash
python -m cProfile -o profile.out app.py
```

The results show that the most time-consuming operations are the image processing filters, especially when applied to large images. The use of a cache significantly improves performance on subsequent runs with the same images and parameters.
