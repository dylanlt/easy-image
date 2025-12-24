from smolagents import CodeAgent, InferenceClientModel, tool
from transformers import pipeline
from PIL import Image, ImageChops
import torch
import numpy as np
import os
import sys

# Define custom tools
@tool
def detect_objects(image_path: str) -> list:
    """
    Detect objects in an image.

    Args:
        image_path: The path to the image file to analyze.
    """
    detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    results = detector(image_path)
    return results

@tool
def segment_image(image_path: str, prompt: str) -> str:
    """
    Segment objects using OWL-ViT for zero-shot detection and Mobile-SAM for high-precision masking.
    Based on Lesson 3: Object Detection.

    Args:
        image_path: The path to the image file to segment.
        prompt: The specific object type you want to mask (e.g., 'bear', 'car').
    """
    from ultralytics import SAM
    from transformers import pipeline
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    raw_image = Image.open(image_path).convert("RGB")
    
    # Consistent token for all model loads from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. Using local execution for tools if possible.")
    
    # 1. Bounding boxes with OWL-ViT
    print(f"DEBUG: Detecting '{prompt}' using OWL-ViT...")
    OWL_checkpoint = "google/owlvit-base-patch32"
    
    # Suppress the meta-parameter warning by not using device if manually moving (or just ignore it)
    # But pipeline handles it better if device is passed correctly.
    detector = pipeline(
        model=OWL_checkpoint,
        task="zero-shot-object-detection",
        device=device,
        token=hf_token
    )
    
    output = detector(
        raw_image,
        candidate_labels=[prompt]
    )
    
    if not output:
        print(f"ERROR: No '{prompt}' detected by OWL-ViT. Terminating.")
        sys.exit(1)
        
    # Extract boxes for SAM
    # OWL-ViT output format: [{'score': ..., 'label': ..., 'box': {'xmin':..., 'ymin':..., 'xmax':..., 'ymax':...}}]
    input_boxes = []
    for detection in output:
        box = detection['box']
        input_boxes.append([box['xmin'], box['ymin'], box['xmax'], box['ymax']])
    
    input_boxes = np.array(input_boxes)
    
    # 2. Segmentation with Mobile SAM
    print(f"DEBUG: Segmenting {len(input_boxes)} detected instances with Mobile SAM...")
    # Ultralytics will download this if not present
    model = SAM("mobile_sam.pt")
    
    # Labels for SAM (1 = positive)
    labels = np.repeat(1, len(input_boxes))
    
    # Run prediction
    # Note: Ultralytics SAM predict expects boxes in [xmin, ymin, xmax, ymax]
    result = model.predict(
        raw_image,
        bboxes=input_boxes,
        labels=labels,
        device=device
    )
    
    if not result or len(result[0].masks.data) == 0:
        print(f"ERROR: SAM failed to generate masks for detected '{prompt}'. Terminating.")
        sys.exit(1)
        
    masks = result[0].masks.data.cpu().numpy() # [N, H, W]
    
    # Combine masks
    total_mask_np = np.zeros(masks[0].shape)
    for mask in masks:
        total_mask_np = np.add(total_mask_np, mask)
        
    # Convert to PIL Image
    # Value > 0 becomes 255 (white), else 0 (black)
    mask_binary = (total_mask_np > 0).astype(np.uint8) * 255
    final_mask = Image.fromarray(mask_binary, mode='L')

    output_path = "mask.png"
    final_mask.save(output_path)
    
    print(f"\n" + "="*40)
    print(f"[VERIFICATION STEP]")
    print(f"Object target: {prompt}")
    print(f"Method: OWL-ViT + Mobile SAM (Lesson 3)")
    print(f"Mask saved to: {os.path.abspath(output_path)}")
    print(f"Instructions: Open mask.png. WHITE areas are the precise silhouettes to be replaced.")
    print(f"="*40)
    
    sys.stdout.flush()
    input("Check the mask.png file. If it's precise, press Enter to continue image generation...")
    
    return output_path

@tool
def generate_image(prompt: str, image_path: str, mask_path: str = None) -> str:
    """
    Generate or edit image using Stable Diffusion Inpainting with Lesson 4 enhancements (higher quality).

    Args:
        prompt: The descriptive prompt for image generation.
        image_path: The path to the original image.
        mask_path: Optional path to a mask image for inpainting.
    """
    from diffusers import StableDiffusionInpaintPipeline
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Using a more reliable model ID for public access
    model_id = "runwayml/stable-diffusion-inpainting"
    hf_token = os.getenv("HF_TOKEN")
    
    # Photorealistic enhancement from Lesson 4
    # Added focus on texture and lighting for "much higher quality"
    enhanced_prompt = f"{prompt}, highly detailed, cinematic lighting, 8k, professional photography, masterpiece, sharp focus, ultra-detailed textures"
    negative_prompt = "cartoon, drawing, anime, blurry, low quality, distorted, watermark, signature, ugly, out of frame, lowres, text, error, cropped, worst quality, jpeg artifacts"
    
    print(f"DEBUG: Generating image with prompt: '{enhanced_prompt}'")
    
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "mps" else torch.float16,
            token=hf_token
        ).to(device)
    except Exception as e:
        print(f"Warning: Failed to load {model_id}: {e}")
        # Final fallback check
        return "Failed to generate image"

    # Open original image
    original_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_image.size
    
    # Resize for model (SD 2 is better at 512x512)
    init_image = original_image.resize((512, 512))
    
    if mask_path and os.path.exists(mask_path):
        mask_image = Image.open(mask_path).convert("RGB").resize((512, 512))
        
        # Lesson 4 Parameters: higher guidance and steps
        # Increased steps to 80 for even higher quality as requested
        output = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=80,
            guidance_scale=12.5,
            strength=0.99 
        ).images[0]
    else:
        # Full image generation
        output = pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            mask_image=Image.new("RGB", (512, 512), (255, 255, 255)),
            num_inference_steps=80,
            guidance_scale=12.5
        ).images[0]

    # Resize back to original aspect ratio
    output = output.resize((orig_w, orig_h))
    
    output_path = "generated.png"
    output.save(output_path)
    print(f"DEBUG: Saved generated image to {output_path}")
    return output_path

# Create agent
agent = CodeAgent(
    tools=[detect_objects, segment_image, generate_image],
    model=InferenceClientModel(
        "meta-llama/Llama-3.3-70B-Instruct",
        token=os.getenv("HF_TOKEN")
    )
)

# Check for image before running
image_path = "image.png"
if not os.path.exists(image_path):
    print(f"Error: {image_path} not found. Please provide an image to proceed.")
    sys.exit(1)

# Get prompt from user
print(f"\n--- Easy Image Agent ---")
print(f"Active image: {image_path}")
user_prompt = input("What would you like me to do with this image? ")

# Ensure the prompt contains the image path for context if not already present
final_prompt = user_prompt
if image_path not in user_prompt:
    final_prompt = f"Using {image_path}: {user_prompt}"

# Use it
result = agent.run(final_prompt)