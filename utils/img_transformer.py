import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("FL-ImageTransformer")

class CANImageTransformer:
    
    def __init__(self, img_size=64):
        self.window_size = 27 
        self.img_size = img_size
        self.channels = 3 
        logger.info(f"Initialized CAN-to-Image Transformer (Paper Method): img_size={img_size}")

    def transform(self, sequence_of_9_features):
        if not isinstance(sequence_of_9_features, np.ndarray):
            feature_matrix = np.array(sequence_of_9_features)
        else:
            feature_matrix = sequence_of_9_features

        if feature_matrix.shape != (self.window_size, 9):
            logger.warning(f"Input sequence has wrong shape. Expected ({self.window_size}, 9), got {feature_matrix.shape}. Skipping.")
            return Image.new('RGB', (self.img_size, self.img_size), color=0)

        normalized_matrix = feature_matrix.copy().astype(np.float32)

        normalized_matrix[:, 0] = (normalized_matrix[:, 0] / 2047.0) * 255.0

        normalized_matrix[:, 1:] = np.clip(normalized_matrix[:, 1:], 0, 255)

        flat_vector = normalized_matrix.flatten()

        image_array_9x9 = flat_vector.reshape((9, 9, self.channels)).astype(np.uint8)

        img_9x9 = Image.fromarray(image_array_9x9, 'RGB')

        img_resized = img_9x9.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            
        return img_resized