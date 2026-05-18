---
title: "🕰️ TalkiePress: Modern News in 1930"
emoji: 📰
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.15.0
python_version: 3.10.13
app_file: app.py
pinned: false
---

# 1930s Newspaper Generator (TalkiePress)

This application generates 1930s-style newspaper articles based on modern news. It directly loads a specialized LLM (`talkie-lm/talkie-1930-13b-base`) that possesses knowledge exclusively from 1930 and earlier. It converts modern events into 1930s journalism, perfectly reflecting the historical background, unique phrasing, and ethical standards of the era.

## Overview
When a user inputs "Current News" (e.g., "The evolution of AI") as a short factual statement, the system abstracts it into a 1930s context (e.g., "The invention of a new calculating machine"). This abstracted event is then passed to the loaded model, which automatically generates a newspaper article written in the authentic style of the 1930s.

This application is highly useful as a high-quality asset generation pipeline for game flavor text or as a dynamic storytelling engine for historical simulations.

## How to Run

### Running on Hugging Face Spaces
Simply upload (or push) the files in this repository to Hugging Face Spaces (Gradio SDK), and the environment will be automatically built and launched.
The model will be automatically downloaded and loaded during the first text generation request.

*※ Hardware Requirements for Spaces:*
To achieve a perfect 1930s reproduction, this app uses a 13B parameter model (`talkie-lm/talkie-1930-13b-base`) trained exclusively on pre-1930s knowledge. To run a model of this size, you **must upgrade the hardware** to a GPU instance (e.g., L4, A10G) from the Settings tab in Hugging Face Spaces. Please note that the application will likely crash due to out-of-memory errors on the free CPU tier (16GB RAM).

### Running Locally
1. Install the required packages:
```bash
pip install -r requirements.txt
```
2. Launch the application:
```bash
python app.py
```
3. Access `http://localhost:7860` in your browser.
If you run this locally on a machine equipped with a GPU (CUDA/MPS), the application will automatically utilize the GPU for inference.
