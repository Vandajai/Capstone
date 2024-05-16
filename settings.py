from pathlib import Path

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file (root directory of the project)
ROOT = FILE.parent

# Sources
IMAGE = 'Image'

SOURCES_LIST = [IMAGE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'Waste_bin.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'Waste_detected.jpg'

# Define relative paths for model files
MODEL_DIR = ROOT 
CHECKPOINT_PATH = MODEL_DIR / 'model_final_1080.pth'
CONFIG_FILE_PATH = MODEL_DIR / 'config_1080.yaml'
TRAIN_DATA_SET_NAME = MODEL_DIR / 'annotations.coco.json'
DETECTION_MODEL = MODEL_DIR / 'your_model_file.extension'

# Convert paths to strings if necessary
CHECKPOINT_PATH = str(CHECKPOINT_PATH)
CONFIG_FILE_PATH = str(CONFIG_FILE_PATH)
TRAIN_DATA_SET_NAME = str(TRAIN_DATA_SET_NAME)
