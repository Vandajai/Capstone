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

def calculate_pixel_counts(outputs, category_names):
    instances = outputs["instances"].to("cpu")
    total_pixels = 0
    category_pixel_counts = {category: 0 for category in category_names}

    for i in range(len(instances)):
        pred_class = instances.pred_classes[i].item()
        mask = instances.pred_masks[i].numpy()
        mask_pixels = np.sum(mask)
        total_pixels += mask_pixels
        category_name = category_names[pred_class]
        category_pixel_counts[category_name] += mask_pixels

    return total_pixels, category_pixel_counts

def generate_summary(outputs, category_names):
    total_pixels, category_pixel_counts = calculate_pixel_counts(outputs, category_names)
    percentages = {cat: (count / total_pixels) * 100 for cat, count in category_pixel_counts.items() if count > 0}
    return total_pixels, category_pixel_counts, percentages

def main():
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

    # Load Pre-trained ML Model
    try:
        model = load_model()
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {CHECKPOINT_PATH}")
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
                        output_image, outputs = infer_image(np.array(uploaded_image), predictor)
                        st.image(output_image, caption='Detected Image',
                                 use_column_width=True)

                        total_pixels, category_pixel_counts, percentages = generate_summary(outputs, [cat["name"] for cat in categories])

                        st.write(f"Total Pixels: {total_pixels}")
                        st.write("Category Pixel Counts:")
                        st.write({cat: count for cat, count in category_pixel_counts.items() if count > 0})
                        st.write("Percentages:")
                        st.write({cat: perc for cat, perc in percentages.items() if perc > 0})

                        with st.expander("Detection Results"):
                            instances = outputs["instances"].to("cpu")
                            for i in range(len(instances)):
                                box = instances.pred_boxes[i].tensor.numpy()[0]
                                area = instances.pred_masks[i].sum()
                                st.write(f"Box: {box}")
                                st.write(f"Area: {area} pixels")
                                
                        st.balloons()
                    except Exception as ex:
                        st.error("Error occurred during prediction.")
                        st.error(ex)
    else:
        st.error("Please select a valid source type!")

if __name__ == '__main__':
    main()
