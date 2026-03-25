from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
def backgrdrmv(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    

    # Step 2: Create a white background (same size)
    white_bg = Image.new("RGB", img.size, (255, 255, 255))

    # Step 3: Paste the image onto the white background
    white_bg.paste(img, (0, 0))

    # Step 4: Resize to target size
    final_img = white_bg.resize((128, 128))

    # Step 5: Convert to array
    img_array = img_to_array(final_img)

    # Step 6: Normalize
    img_array = img_array / 255.0

    # Step 7: Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
