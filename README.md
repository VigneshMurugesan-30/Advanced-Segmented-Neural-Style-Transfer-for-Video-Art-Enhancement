# Advanced-Segmented-Neural-Style-Transfer-for-Video-Art-Enhancement

## Overview

This project focuses on applying deep learning techniques, specifically **Neural Style Transfer**, for signal and image processing tasks. The project demonstrates the use of advanced architectures like **VGG-19** to achieve high-quality style transfer in images. By utilizing **Gram matrices** to represent the style of an image, this project creates visually appealing combinations of content and artistic styles.

## Key Concepts

### Neural Style Transfer
Neural Style Transfer (NST) is the process of blending the style of one image (typically an artwork) with the content of another image (a photograph), generating a new image that maintains the core structure of the content while adopting the visual patterns of the style image.

### VGG-19 Architecture
VGG-19 is a convolutional neural network that processes fixed-size RGB images of shape (224, 224, 3). Some of the key features of VGG-19 include:
- **3x3 kernels** with a stride of 1 pixel.
- **Max pooling** layers with 2x2 windows and stride 2.
- Use of **ReLU activation** for non-linearity, improving performance over previous activation functions like tanh or sigmoid.
- Three fully connected layers, with the final softmax layer used for classification.

### Gram Matrix
A **Gram matrix** is used to capture the style of an image. It is constructed from the feature maps of an image after passing through a convolutional neural network like VGG-19. The Gram matrix helps to measure the correlations between different feature maps, capturing the texture and style of the image.

## Project Objectives

1. **Apply Neural Style Transfer**: Utilize the VGG-19 network to perform style transfer by extracting content and style representations from images.
2. **Leverage Gram Matrices**: Construct Gram matrices to encode style information and apply this style to content images.
3. **Understand VGG-19**: Explore the VGG-19 architecture for its feature extraction capabilities in style transfer tasks.

## Implementation Details

- **Input Size**: The VGG-19 model requires input images of size (224x224) pixels.
- **Feature Extraction**: Feature maps are extracted from various layers of VGG-19 to represent the content and style of the images.
- **Style Representation**: The Gram matrix is used to model the style, ensuring the transfer of textures and patterns from the style image to the content image.
- **Optimization**: The style transfer process is optimized to balance content preservation and style application.

## Methodology

1. **Preprocessing**: Input images are resized to 224x224 pixels, and the mean RGB value is subtracted from each pixel.
2. **Feature Space Construction**: Feature maps from the VGG-19 layers are used to construct the Gram matrix.
3. **Loss Functions**: The loss is computed by balancing content loss (which ensures the content structure is preserved) and style loss (which ensures the style is accurately transferred).
4. **Training**: The model is trained using backpropagation to minimize both content and style losses.

## Results

The project demonstrates successful application of neural style transfer, creating images that maintain the content of a source image while adopting the visual style of another image. By using Gram matrices to capture and apply style information, the results are visually coherent and artistically appealing.

## Tools and Technologies

- **Python**
- **TensorFlow / PyTorch**
- **VGG-19 Pre-trained Model**
- **Gram Matrix Calculations**

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VigneshMurugesan-30/Advanced-Segmented-Neural-Style-Transfer-for-Video-Art-Enhancement.git
   cd deep-learning-image-processing
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Run the style transfer script**:
   ```bash
   python style_transfer.py --content-path <content_image> --style-path <style_image>

## References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.

## Contact
For more details, feel free to reach out at vigneshmurugesan309@gmail.com

