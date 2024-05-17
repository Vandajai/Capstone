import streamlit as st
import numpy as np
import pandas as pd
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pathlib import Path
import PIL

# Local Modules
import settings

# Define the paths
CHECKPOINT_PATH = settings.CHECKPOINT_PATH
CONFIG_FILE_PATH = settings.CONFIG_FILE_PATH
TRAIN_DATA_SET_NAME = settings.TRAIN_DATA_SET_NAME

cfg = get_cfg()
cfg.merge_from_file(CONFIG_FILE_PATH)
cfg.MODEL.WEIGHTS = CHECKPOINT_PATH
cfg.DATASETS.TRAIN = TRAIN_DATA_SET_NAME
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.DEVICE = "cpu"  # cpu or cuda

# Define categories directly
categories = [
    {"id": 0, "name": "BG-uSun-feAy", "supercategory": "none"},
    {"id": 1, "name": "Disposable", "supercategory": "BG-uSun-feAy"},
    {"id": 2, "name": "Garbage", "supercategory": "BG-uSun-feAy"},
    {"id": 3, "name": "Glass_Item", "supercategory": "BG-uSun-feAy"},
    {"id": 4, "name": "Metal", "supercategory": "BG-uSun-feAy"},
    {"id": 5, "name": "Other Contamination", "supercategory": "BG-uSun-feAy"},
    {"id": 6, "name": "Plastic", "supercategory": "BG-uSun-feAy"},
    {"id": 7, "name": "Plastic_Bag", "supercategory": "BG-uSun-feAy"},
]

# Register categories in the MetadataCatalog
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).set(thing_classes=[cat["name"] for cat in categories])

def load_model():
    return DefaultPredictor(cfg)

def infer_image(image, predictor):
    outputs = predictor(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1], outputs

def calculate_percentages_and_areas(outputs, category_names, pixel_to_cm):
    instances = outputs["instances"].to("cpu")
    total_pixels = sum(instances.pred_masks.numpy().sum(axis=(1, 2)))
    category_pixel_counts = {category: 0 for category in category_names}
    category_areas_cm2 = {category: 0 for category in category_names}

    for i in range(len(instances)):
        pred_class = instances.pred_classes[i].item()
        mask = instances.pred_masks[i].numpy()
        mask_pixels = np.sum(mask)
        category_name = category_names[pred_class]
        category_pixel_counts[category_name] += mask_pixels

        # Calculate area in cm²
        category_areas_cm2[category_name] += mask_pixels * (pixel_to_cm ** 2)

    # Calculate percentages relative to the "Garbage" area
    garbage_area = category_pixel_counts["Garbage"]
    percentages = {cat: (count / garbage_area) * 100 for cat, count in category_pixel_counts.items() if count > 0}
    areas_cm2 = {cat: area for cat, area in category_areas_cm2.items() if area > 0}

    # Round percentages and append '%'
    percentages = {cat: f"{round(perc)}%" for cat, perc in percentages.items()}

    return percentages, areas_cm2

def main():
    st.markdown(
        '<p style="text-align:center; color:red; font-size:30px;">Capstone Project</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center; font-size:35px; font-weight:bold;">Instance Segmentation using Mask R-CNN</p>',
        unsafe_allow_html=True
    )

    st.sidebar.header("ML Model Config")

    # Model Options: Detection or Segmentation
    model_type = st.sidebar.selectbox(
        "Select Task", ['Detection', 'Segmentation'])

    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

    try:
        model = load_model()
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {CHECKPOINT_PATH}")
        st.error(ex)

    st.sidebar.header("Image Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png"))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    st.image(default_image_path, caption="Default Image", use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
            else:
                if st.sidebar.button('Detect Waste'):
                    try:
                        predictor = load_model()
                        output_image, outputs = infer_image(np.array(uploaded_image), predictor)
                        st.image(output_image, caption='Detected Image', use_column_width=True)

                        pixel_to_cm = 0.026  # Example conversion factor, adjust as needed
                        percentages, areas_cm2 = calculate_percentages_and_areas(outputs, [cat["name"] for cat in categories], pixel_to_cm)

                        data = {
                            "Category": list(percentages.keys()),
                            "Percentage (%)": list(percentages.values()),
                            "Area (cm²)": [areas_cm2[cat] for cat in percentages.keys()]
                        }

                        df = pd.DataFrame(data)

                        with st.expander("Detection Results"):
                            st.table(df.sort_values(by='Category', key=lambda col: col.str.lower() != 'garbage'))

                        st.balloons()
                    except Exception as ex:
                        st.error("Error occurred during prediction.")
                        st.error(ex)
    else:
        st.error("Please select a valid source type!")

if __name__ == '__main__':
    main()
