# Facial Expression Recognition Using Multi-Modal Deep Learning with Graph Attention Networks

This repository contains code for a **real-time facial expression recognition system** that fuses geometric and appearance features using CNNs and Graph Attention Networks (GATs), plus cross-modal fusion and temporal smoothing for improved performance on video.

---

## 🚩 Overview

- **CNN**: Extracts appearance-based features (texture/emotion cues) from face crops.
- **GAT**: Models facial landmark relations, capturing structure and spatial dynamics.
- **Fusion**: Merges modalities for accuracy and robustness.
- **Temporal Smoothing**: Makes predictions stable in video streams.
- **Webcam Demo**: Real-time emotion recognition via camera.

---

## 🚀 Installation

1. **Clone the repository**
   git clone  https://github.com/vighneshb04/Facial-Expression-Recognition-GAN
   
   cd facial-expression-recognition

3. **Set up a Python virtual environment**

4. **Install required dependencies**
 
   pip install --upgrade pip
   
   pip install -r requirements.txt
---

> **Note:**  
> - You may need to install [Dlib](http://dlib.net/) and [PyTorch](https://pytorch.org/) separately if you encounter issues with those packages (refer to their official documentation).
> - Make sure you have Python 3.7 or newer installed.
> - For GPU acceleration, ensure your system has the proper CUDA drivers and install the appropriate PyTorch version.

## 🧠 Model Architecture

- **CNN Stream**: Learns textures/emotions from images.
- **GAT Stream**: Learns spatial/structural relations from 68-point Dlib landmarks.
- **Feature Fusion**: Concatenates or attends to both streams.
- **Temporal Smoothing**: Exponential moving average enhances video stability.

**Pipeline:**  
Input video → Frame extraction → Landmark detection (Dlib)  
→ [Landmark Graph → GAT] + [Cropped face image → CNN]  
→ Fusion → Temporal smoothing → Emotion prediction

---

## 📊 Datasets

- **CREMA-D**: 7,442 video clips, 6 emotion classes, 68-point landmarks.
- **Extracted Frames**: ~569,000 images with landmark annotations.

---

## 🏅 Results

| Model                       | Accuracy (%) |
|-----------------------------|:-----------:|
| CNN Only                    | 38.20       |
| GAT Only                    | 42.10       |
| CNN + GAT                   | 45.05       |
| CNN + GAT + Temporal Smoothing | **52.88** |

- Confusion (fear/neutral) reduced by 71% with full pipeline.
- Real-time performance (~12 FPS on consumer hardware).

---

## 📚 References

Please cite the accompanying paper if using this code in your research.  
References to related and foundational works are in `Facial_Expression_Recognition_Using_Multi-Model_Deep_Learning_with_Graph_Attention_Networks.pdf`.

---

## ⚖️ License

MIT License.

---

## ✨ Acknowledgements

- Built using PyTorch, Dlib, and standard deep learning toolkits.
- Thanks to [CREMA-D authors](https://github.com/CheyneyComputerScience/CREMA-D/) and the open-source community.

---

**For questions or contributions, open an issue or pull request!**
