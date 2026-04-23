---
title: MobileDDD
emoji: 🏢
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 5.38.1
python_version: 3.10.0
app_file: app.py
pinned: false
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference  

# MobileDDD: Driver Drowsiness Detection

Driver Drowsiness Detection using a fine‑tuned **MobileNetV2** model, deployed on **Hugging Face Spaces** with a Gradio interface.

**Live Demo**: [Try it on Hugging Face](https://huggingface.co/spaces/Mielle85/MobileDDD)

---

## Overview
This project demonstrates how deep learning can be applied to detect driver drowsiness from facial images.  
The model is based on **MobileNetV2**, fine‑tuned on a custom dataset, and integrated into a Gradio app for easy interaction.

---

## Features
- Fine‑tuned MobileNetV2 for drowsiness classification  
- Interactive Gradio web app  
- Hosted on Hugging Face Spaces  
- Model weights available in Hugging Face Hub (`Mielle85/mobileDDD`)  

---

## Installation (Run Locally)
Clone the repo and install dependencies:
```bash
git clone https://github.com/Miola585/MobileDDD.git
cd MobileDDD
pip install -r requirements.txt
python app.py
