# 🎨 Easy Image Agent

A powerful, agentic AI tool for precise image manipulation. This agent combines state-of-the-art computer vision models for detection, segmentation, and inpainting to transform images based on simple natural language prompts.

## 🚀 Overview

The **Easy Image Agent** follows a sophisticated multi-step pipeline:
1.  **Zero-Shot Detection**: Uses **OWL-ViT** to identify objects in your image based on your text prompt.
2.  **Precision Segmentation**: Leverages **Mobile SAM** (Segment Anything Model) to create high-fidelity silhouettes of the detected objects.
3.  **Advanced Inpainting**: Employs **Stable Diffusion** with Lesson 4 enhancements (high inference steps, cinematic guidance) to replace the masked areas with hyper-realistic results.

---

## 🛠️ Key Features

- **Precise Silhouettes**: Unlike simple bounding-box masking, SAM ensures only the object is replaced, perfectly preserving the background.
- **Visual Verification**: The agent pauses after generating a mask, allowing you to verify the selection before proceeding with expensive generation.
- **Hardware Accelerated**: Optimized for **Apple Silicon (MPS)** and NVIDIA GPUs.
- **Photorealistic Quality**: Built-in cinematic lighting and masterpiece-level prompt enhancements.

---

## 📋 Prerequisites

- **Python**: 3.10 or higher.
- **Hardware**: Apple Silicon Mac (M1/M2/M3) or NVIDIA GPU recommended.
- **API Key**: A Hugging Face token with access to the required models.

---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd easy-image-agent
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Export your API Token**:
   ```bash
   export HF_TOKEN="your_huggingface_token_here"
   ```

2. **Prepare your image**:
   Place an image named `image.png` in the root directory.

3. **Run the Agent**:
   ```bash
   python easy-image-agent.py
   ```

4. **Interact**:
   When prompted, tell the agent what you'd like to do.
   *Example: "Replace the bears with capybaras"*

---

## 📂 Project Structure

- `easy-image-agent.py`: The main agentic script.
- `requirements.txt`: Project dependencies.
- `mask.png`: The generated mask (created during runtime).
- `generated.png`: The final result.

---

## 🧠 Educational Context

This project is inspired by https://learn.deeplearning.ai/courses/prompt-engineering-for-vision-models/lesson/q0uu0/introduction
- **Lesson 3**: Advanced object detection and segmentation pipelines.
- **Lesson 4**: Mastering Stable Diffusion hyperparameters for high-quality inpainting.
