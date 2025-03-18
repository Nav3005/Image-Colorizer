# Image Colorizer

This Streamlit application uses deep learning to automatically colorize black and white images. The app employs a pre-trained model based on the Zhang et al. colorization architecture to add realistic colors to grayscale photographs.

## Features

- Upload black and white images in JPG, JPEG, or PNG format
- Real-time colorization using deep learning
- Download capability for colorized images
- User-friendly web interface

## Prerequisites

- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)
- Model files (download instructions below)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Nav3005/Image-Colorizer.git
cd Image-Colorizer
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the model files:
   - Create a `Model` directory in the project root
   - Download the following files and place them in the `Model` directory:
     - [colorization_deploy_v2.prototxt](https://raw.githubusercontent.com/richzhang/colorization/master/colorization/models/colorization_deploy_v2.prototxt)
     - [colorization_release_v2.caffemodel](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1)
     - [pts_in_hull.npy](https://raw.githubusercontent.com/richzhang/colorization/master/colorization/resources/pts_in_hull.npy)

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a black and white image using the file uploader

4. Wait for the colorization process to complete

5. Download the colorized image using the "Download Colorized Image" button

## Model Information

The colorization model used in this application is based on the research paper "Colorful Image Colorization" by Zhang et al. It uses a deep neural network trained on a large dataset of color images to predict the most likely colors for a given grayscale input.

## Limitations

- The model works best with natural photographs
- Results may vary depending on the quality and content of the input image
- Processing time depends on the image size and your computer's specifications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original colorization model by [Richard Zhang](https://github.com/richzhang/colorization)
- Built with Streamlit 