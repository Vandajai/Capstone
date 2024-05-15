
# <span style="color:deepskyblue"> Real-time waste Detection with Mask-RCNN & Streamlit </span>
<img src=".github/Detectron2-Logo-Horz.svg" width="300" >

<a href="https://opensource.facebook.com/support-ukraine">
  <img src="https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB" alt="Support Ukraine - Help Provide Humanitarian Aid to Ukraine." />
</a>

Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.

<div align="center">
  <img src="https://github.com/Vandajai/Capstone_project1/blob/main/images/Waste_detected.jpg"/>
</div>
<br>



## Installation

See [installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- Clone the repository: git clone <https://github.com/Vandajai/Capstone.git>
- Change to the repository directory: `cd Capstone_Project1`
- Create `weights` and `images` directories inside the project.
- Download the pre-trained detectron2 weights and save them to the `weights` directory in the same project.

## Getting Started

See [Getting Started with Detectron2](https://detectron2.readthedocs.io/tutorials/getting_started.html),



## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron2 Model Zoo](MODEL_ZOO.md).

## Usage

- Run the app with the following command: `streamlit run app.py`
- The app should open in a new browser window.

### ML Model Config

- Select task (Detection, Segmentation)
- Select model confidence
- Use the slider to adjust the confidence threshold (25-100) for the model.

One the model config is done, select a source.

### Detection on images

- The default image with its objects-detected image is displayed on the main page.
- Select a source. (radio button selection `Image`).
- Upload an image by clicking on the "Browse files" button.
- Click the "Detect waste" button to run the waste detection algorithm on the uploaded image with the selected confidence threshold.
- The resulting image with objects detected will be displayed on the page. 



