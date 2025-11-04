import streamlit as st
import pandas as pd
from pathlib import Path
from src import pipeline, io_utils, ml

st.title("PhotoLab Express")

# --- 1. File Upload ---
uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files to a temporary directory
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    image_paths = []
    for uploaded_file in uploaded_files:
        image_path = temp_dir / uploaded_file.name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(image_path)


    # --- 2. Display Thumbnails ---
    st.image([str(p) for p in image_paths], width=100, caption=[p.name for p in image_paths])

    # --- 3. Filter Selection ---
    filter_options = ["Sobel", "Canny", "Gaussian Blur", "Sharpen", "Random Hue Shift"]
    selected_filters = st.multiselect("Select filters:", filter_options)

    # --- 4. Face Detection ---
    face_detection = st.checkbox("Detect faces")

    # --- 5. Processing ---
    if st.button("Process"):
        with st.spinner("Processing images..."):
            # --- Run Pipeline ---
            results = pipeline.run_pipeline(image_paths, selected_filters, face_detection)

            # --- AI Classification ---
            raw_images = [io_utils.load_image(p) for p in image_paths]
            classifications = ml.classify_batch(raw_images)

            # --- Display Results ---
            st.header("Processed Images")

            images_with_faces = []
            images_without_faces = []
            processed_images_for_saving = []

            for img, faces_detected in results:
                processed_images_for_saving.append(img)
                if faces_detected:
                    images_with_faces.append(img)
                else:
                    images_without_faces.append(img)

            if images_with_faces:
                st.subheader("Images with Detected Faces")
                st.image([io_utils.cv2_to_pil(img) for img in images_with_faces], width=200)

            if images_without_faces:
                st.subheader("Images without Detected Faces")
                st.image([io_utils.cv2_to_pil(img) for img in images_without_faces], width=200)

            # --- Export CSV ---
            st.info("Note: The AI classification is based on the CLIP model by OpenAI. The labels are more accurate but may still produce unexpected results.")
            metrics = {
                "filename": [p.name for p in image_paths],
                "classification": [c[0] for c in classifications],
            }
            df = pd.DataFrame(metrics)
            st.dataframe(df)

            st.download_button(
                label="Download data as CSV",
                data=df.to_csv().encode("utf-8"),
                file_name="photolab_express_metrics.csv",
                mime="text/csv",
            )

            # --- Save filtered images ---
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            for i, img in enumerate(processed_images_for_saving):
                classification = classifications[i][0].replace("a photo of a ", "").replace("an ", "").replace("a ", "")
                category_dir = output_dir / classification
                category_dir.mkdir(exist_ok=True)
                io_utils.save_image(img, category_dir / image_paths[i].name)

            st.success("Processing complete!")
