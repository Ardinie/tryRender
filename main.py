import os
import numpy as np
import torch
from torch import tensor
from torchvision.transforms import ToTensor, Resize
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_dilation, binary_erosion

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello from Render!"}

def apply_pattern_to_shape(pattern_img_tensor, mask_tensor, plain_img_tensor):
    if mask_tensor.shape[0] == 1:  
        mask_tensor = mask_tensor.repeat(3, 1, 1)  
    if plain_img_tensor.shape[0] == 1:  
        plain_img_tensor = plain_img_tensor.repeat(3, 1, 1)

    shaded_pattern = pattern_img_tensor * plain_img_tensor

    patterned_shape = shaded_pattern * mask_tensor + (1 - mask_tensor)
    return patterned_shape

@app.post("/process-fabric/")
async def process_fabric(
    plain_image_choice: int = Query(1, ge=1, le=5, description="Choice of plain image (1, 2, 3,4 or 5)"),
    pattern_image: UploadFile = File(...),
):
    try:
        print(f"Selected plain image choice: {plain_image_choice}")

        # Plain image paths mapping
        plain_image_paths = {
            1: "static/plain_image_1.jpg",
            2: "static/plain_image_3.jpg",
            3: "static/plain_image_2.jpg",
            4: "static/cheongsam.jpg",
            5: "static/Saree.jpg",
        }

        # Validate plain image path
        plain_image_path = plain_image_paths.get(plain_image_choice)
        if not plain_image_path or not os.path.exists(plain_image_path):
            raise HTTPException(status_code=400, detail=f"Invalid plain image choice: {plain_image_choice}")

        print(f"Uploaded pattern image content type: {pattern_image.content_type}")

        # Validate the uploaded pattern image
        if pattern_image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(status_code=400, detail="Invalid pattern image format. Only PNG and JPEG are supported.")
            
        # Load and preprocess images
        plain_img = Image.open(plain_image_path).convert("L") 
        pattern_img = Image.open(pattern_image.file).convert("RGB")

        # Resize images dynamically
        max_size = 256
        resize = Resize((max_size, max_size))
        plain_img_resized = resize(plain_img)
        pattern_img_resized = ToTensor()(resize(pattern_img))

        # Normalize grayscale values for plain image tensor
        plain_img_tensor = ToTensor()(plain_img_resized)

        # Create binary mask
        threshold = 200
        binary_mask = np.array(plain_img_resized) < threshold
        filled_mask = binary_fill_holes(binary_mask).astype(np.float32)

        # Convert mask to tensor
        mask_tensor = torch.tensor(filled_mask).unsqueeze(0)

        # Apply pattern to shape
        patterned_tensor = apply_pattern_to_shape(pattern_img_resized, mask_tensor, plain_img_tensor)

        # Generate an outline using dilation and erosion
        dilated_mask = binary_dilation(filled_mask)
        eroded_mask = binary_erosion(filled_mask)
        outline = dilated_mask.astype(float) - eroded_mask.astype(float)

        # Convert outline to tensor
        outline_tensor = torch.tensor(outline).unsqueeze(0).repeat(3, 1, 1)
        black_outline = torch.zeros_like(outline_tensor)

        # Combine pattern with the black outline
        final_tensor = patterned_tensor * (1 - outline_tensor) + black_outline * outline_tensor

        # Convert tensor to image
        final_image_array = (final_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        final_image = Image.fromarray(final_image_array)


    except HTTPException as http_error:
        print(f"HTTP error: {http_error.detail}")
        raise http_error
    except Exception as e:
        import traceback
        print("Error traceback:", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")
