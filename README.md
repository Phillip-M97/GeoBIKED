# GeoBIKED
This dataset and code are presented in the paper GeoBIKED: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design.

This work was presented at the Modeling, Data Analytics and AI in Engineering Conference (MadeAI), held in Porto, Portugal in July 2024 https://madeai-eng.org/

## License
This code is published under the MIT license. Feel free to use all portions for research or related projects under the prerequisite to cite the paper "Phillip Mueller, Sebastian Mueller and Lars Mikelsons. "GeoBIKED: A Dataset with Geometric Features and Automated Labeling Techniques to Enable Deep Generative Models in Engineering Design." 

## Downloading the Data
Due to file size constraints, download the data from [Google Drive](https://drive.google.com/drive/folders/1s2gILDboW2S66XxS2TtkOdsEYOMqe1TH?usp=sharing).
To work with it, download and unzip the data and then select the required folders.

# Description of the Dataset
**GeoBiked_parameters.csv** contains the annotations and features associated with the data. The first column *Bike index* contains the index of the samples that can be matched with the image indices. The csv-file contains the geometric reference points and other features such as Bike Styles, Frame Sizes, RIM Styles, ...
## Geometrical Data
The subsequent columns contain the geometrical annotations. **We provide the data in millimeters**. Columns two and three (*x_zero* and *y_zero*) describe the center of the local coordinate system of each bike within the image. We chose the rear-wheel-center as the center of the local coordinate system. All other points are relative to this local coordinate system.
Keep in mind that the data needs to be scaled according to the image resolution to draw geometric correspondences. The scaling factors are:
- (256 x 256) --> 1 px = 10,19mm
- (2048 x 2048) --> 1 px = 1,27mm

The geometric features are essentially a low-dimensional structured point-cloud, describing the geometrical layout of each bicycle through the position of characteristic intersections and geometric points.
![geo_points](https://github.com/Phillip-M97/GeoBIKED/assets/86968936/b6692fcd-45f7-4780-8c54-c15588428ed7)

## Features
**Bike Styles** contains strings describing the style of the bicycles. We provide 19 different styles.
![Style_Dist](https://github.com/Phillip-M97/GeoBIKED/assets/86968936/247eb70c-81b0-4f25-8c32-fb0e8ed3441e)

We also provide information about the **RIM Style** of each bike as well as the **Fork Type**.
**Tube Sizes**,**Frame Sizes** and the number of teeth on the **Chainring** are associated technical features.
![Tube_and_Fork](https://github.com/Phillip-M97/GeoBIKED/assets/86968936/830f5b70-1cfa-4919-a8b0-fdc7231424f0)


# Detecting Geometric Correspondences with Diffusion Hyperfeatures
This information corresponds to the task of automatic dataset annotation with geometric reference points by passing the model a handful of manually annotated examples.
The relevant code is in the folder *Point_Detection_Hyperfeatures*.

The code for automatic annotation of geometric correspondences (Point Annotation) is based on the paper [Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence](https://arxiv.org/abs/2305.14334) Please also refer to the provided [repository](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures) by the authors. 
In case of troubles with packages and dependencies, we advise to refer to the original repository provided by Luo et al.

## Test Data
The benchmark dataset discussed in the paper is a derived subset from our provided GeoBiked Dataset. It can also be found in the [Drive](https://drive.google.com/drive/folders/1s2gILDboW2S66XxS2TtkOdsEYOMqe1TH?usp=sharing). To use this, or any other dataset, simply download it and place it within the "assets" folder.

# Annotation with Text Descriptions
This corresponds to the section in the paper that discusses the processing of the GeoBiked dataset with pretrained Vision-Language Models to automatically annotate it with creative and controllable text descriptions.

## Vision-Language Models
We evaluate GPT-4o as our Vision-Language Model.

## System Prompts
The system prompts that we used to generate the text descriptions with the different models are provided in the corresponding folder *Text_Annotations_with_VLMs*.
We provide the entire system prompt for each evaluated model as well as some exemplary text descriptions. The descriptions can be matched with the corresponding images by their indices.


