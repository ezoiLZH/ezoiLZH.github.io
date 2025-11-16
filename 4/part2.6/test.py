from complete_nerf_training import load_my_data, create_intrinsic_matrix, render_image
import numpy as np
import cv2
from PIL import Image
import os


frame_path = 'spherical_rendering'

paths = os.listdir(frame_path)
print(paths)


render_images = []
for path in paths:
    if path.endswith('.png'):
        img = cv2.imread(os.path.join(frame_path, path))
        image_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        image_uint8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        render_images.append(image_uint8)

frames_for_gif = [Image.fromarray(frame) for frame in render_images]
frames_for_gif[0].save(
    'spherical_rendering.gif',
    save_all=True,
    append_images=frames_for_gif[1:],
    duration=500,  # milliseconds per frame
    loop=0
)