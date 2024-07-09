# GeoBIKED
This dataset and code are presented in the paper GeoBIKED: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design.

This work was presented at the Modeling, Data Analytics and AI in Engineering Conference (MadeAI), held in Porto, Portugal in July 2024 https://madeai-eng.org/

## License
This code is published under teh MIT license. Feel free to use all portions for research or related projects under the prerequisite to cite the paper "Phillip Mueller, Sebastian Mueller and Lars Mikelsons. "GeoBIKED: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design." 

## Required Packages
- PyTorch > 2.0
- Matplotlib
- skLearn
- Diffusers

The code for automatic annotation of geometric correspondences (Point Annotation) is based on the paper [Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence](https://arxiv.org/abs/2305.14334) Please also refer to the provided [repository](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures) by the authors.

## Downloading the Data
Due to file size constraints, download the data from Google Drive (LINK).
To work with it, download and unzip the data and then select the required folders.

## Description of the Dataset
**GeoBiked_parameters.csv** contains the annotations and features associated with the data. The first column *Bike index* contains the index of the samples that can be matched with the image indices. The csv-file contains the geometric reference points and other features such as Bike Styles, Frame Sizes, RIM Styles, ...
### Geometrical Data
The subsequent columns contain the geometrical annotations. **We provide the data in millimeters**. Columns two and three (*x_zero* and *y_zero*) describe the center of the local coordinate system of each bike within the image. We chose the rear-wheel-center as the center of the local coordinate system. All other points are relative to this local coordinate system.
Keep in mind that the data needs to be scaled according to the image resolution to draw geometric correspondences. The scaling factors are:
- (256 x 256) --> 1 px = 10,19mm
- (2048 x 2048) --> 1 px = 1,27mm

### Features
**Bike Styles** contains strings describing the style of the bicycles. We provide 19 different styles.
![Style_Dist](https://github.com/Phillip-M97/GeoBIKED/assets/86968936/247eb70c-81b0-4f25-8c32-fb0e8ed3441e)


