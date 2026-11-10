import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps

# Paths
project_root = Path('C:/RVU/Sem5/low-light-image-enhancement')
checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
input_image_path = Path(r'C:/Users/medha/.gemini/antigravity/brain/85effbfe-f0ce-457d-b112-d118d94845f9/.user_uploaded/media_1793680623307_a0cb634e.jpg')
output_path = Path(r'C:/Users/medha/.gemini/antigravity/brain/85effbfe-f0ce-457d-b112-d118d94845f9/enhanced_output.jpg')

# Load model (same as app)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(str(project_root))
from model import EnhancementCNN, MODEL_VERSION
model = EnhancementCNN().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
if checkpoint.get('model_version') != MODEL_VERSION:
    raise ValueError('Checkpoint version mismatch')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
image = ImageOps.exif_transpose(Image.open(input_image_path)).convert('RGB')
# Resize to same size used in app (384x256) while preserving aspect
IMAGE_SIZE = (384, 256)
resized = ImageOps.contain(image, IMAGE_SIZE, method=Image.Resampling.BILINEAR)
arr = np.asarray(resized, dtype=np.float32) / 255.0
tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

# Inference
with torch.inference_mode():
    out = model(tensor).clamp(0,1)
out_arr = (out[0].cpu().permute(1,2,0).numpy()*255).astype(np.uint8)
enhanced = Image.fromarray(out_arr)
# Save enhanced image
enhanced.save(output_path)
print('Enhanced image saved to', output_path)

