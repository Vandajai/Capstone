import streamlit as st
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pathlib import Path
import PIL

# Local Modules
import settings
from settings import MODEL_DIR


# Define the paths
# Define the paths using settings.py
CHECKPOINT_PATH = settings.CHECKPOINT_PATH
CONFIG_FILE_PATH = settings.CONFIG_FILE_PATH
TRAIN_DATA_SET_NAME = settings.TRAIN_DATA_SET_NAME
DETECTION_MODEL = settings.DETECTION_MODEL




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
    return out.get_image()[:, :, ::-1]

def main():
    # Subtitle in bold

    # Centered title with red color
    st.markdown(
    '<p style="text-align:center; color:red; font-size:30px;">Capstone Project</p>',
    unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align:center; font-size:35px; font-weight:bold;">Instance Segmentation using Mask R-CNN</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.header("ML Model Config")

    # Model Options
    model_type = st.sidebar.radio(
        "Select Task", ['Detection', 'Segmentation'])

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

    # Selecting Detection Or Segmentation
    if model_type == 'Detection':
        model_path = Path(settings.DETECTION_MODEL)
    elif model_type == 'Segmentation':
        model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = load_model()
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.header("Image Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    source_img = None
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png"))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                             use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                             use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                         use_column_width=True)
            else:
                if st.sidebar.button('Detect Waste'):
                    try:
                        predictor = load_model()
                        output_image = infer_image(np.array(uploaded_image), predictor)
                        st.image(output_image, caption='Detected Image',
                                 use_column_width=True)
                    except Exception as ex:
                        st.error("Error occurred during prediction.")
                        st.error(ex)
    else:
        st.error("Please select a valid source type!")

if __name__ == '__main__':
    main()