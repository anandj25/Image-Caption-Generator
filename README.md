# Image Caption Generator using CNN and LSTM  

This project implements an **Image Caption Generator** using a combination of **Convolutional Neural Networks (CNNs)** for image feature extraction and **Long Short-Term Memory (LSTM)** networks for sequence processing and caption generation. The system is designed to create textual descriptions of images, providing significant accessibility benefits for visually impaired individuals.

---

## Table of Contents  
- [Overview](#overview)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Tools and Technologies](#tools-and-technologies)  
- [Results](#results)  
- [Future Work](#future-work)  
  

---

## Overview  

Over 36 million individuals are blind, and 217 million have moderate to severe visual impairment (MSVI). This project aims to make digital content more accessible to these users by providing automatic verbal descriptions of images. Our system is trained on the **COCO 2017 dataset** and combines the capabilities of CNNs, LSTMs, and Natural Language Processing (NLP).  

By bridging computer vision and NLP, this project contributes to advancing inclusivity in the digital age.  

---

## Dataset  

We used the **COCO 2017 dataset** for training and validation.  
- **Train2017**: 118K images  
- **Val2017**: 5K images  

Key dataset modifications:
- Parsed `.json` annotation files to extract essential metadata (e.g., image dimensions, bounding boxes, captions).  
- Focused on two categories—**elephant** and **sandwich**—to optimize performance with limited computational resources.  

---

## Methodology  

The project implementation follows these steps:  

1. **Text to Token Conversion**  
   - Preprocessing captions: removing punctuation, converting to lowercase, and adding `<start>` and `<end>` tags.  
   - Tokenization to convert text into sequences of integers.  

2. **Feature Extraction**  
   - Used the pre-trained **Xception** model to extract 2048-dimensional feature vectors from images.  
   - Resized images to 299x299 and normalized for compatibility with the model.  

3. **Data Generator**  
   - Generated input-output pairs for training by combining image features with partial captions.  

4. **CNN-LSTM Model**  
   - Combined CNNs (image feature extraction) with LSTMs (sequence generation) to predict captions.  
   - Used activation functions like ReLU and SoftMax and optimized the model with Adam optimizer.  

5. **Evaluation**  
   - Assessed generated captions using custom test data to measure accuracy and contextual relevance.  

---

## Tools and Technologies  

- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, Matplotlib, Scikit-learn  
- **Models**: Pre-trained **Xception** for feature extraction, custom CNN-LSTM architecture  
- **Environment**: Jupyter Notebook  

---

## Results  

### Sample Outputs  
1. **Correct Prediction**  
   - Input Image: A sandwich  
   - Generated Caption: "A sandwich on a plate."  

2. **Misclassified Cases**  
   - Input Image: An elephant misclassified as a giraffe due to limited training categories.  

Despite constraints, the model demonstrated promising results, and accuracy is expected to improve with additional training data and categories.  

---

## Future Work  

1. **Text-to-Speech (TTS) API**  
   - Integrate TTS for verbalizing captions to assist visually impaired users.  

2. **Broader Dataset**  
   - Expand categories to enhance model versatility.  

3. **Practical Applications**  
   - Use in education and safety-critical environments, enabling better accessibility for all users.  



