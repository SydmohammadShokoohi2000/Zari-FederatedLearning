import logging
import numpy as np
import tensorflow as tf

from utils.model import ZariTFLite
from utils.img_transformer import CANImageTransformer 

logger = logging.getLogger("FL-Aggregator")

class TFLiteAggregator:
    
    def __init__(self, **kwargs):
        self.model = None
        self.args = kwargs
        self.test_global = None
        self.image_transformer = None
        
        if self.args.get("use_image_input", False):
            self.image_transformer = CANImageTransformer(
                img_size=self.args.get("img_size"),
                channels=self.args.get("channels")
            )
        
        logger.info("TFLiteAggregator initialized.")

    def initialize_global_model(self):
        logger.info("Initializing global model in aggregator...")
        zari_model_wrapper = ZariTFLite(
            input_size=self.args.get("input_size"),
            num_classes=self.args.get("num_classes", 2),
            use_image_input=self.args.get("use_image_input", True),
            resource_profile="high",
            img_height=self.args.get("img_size"),
            img_width=self.args.get("img_size"),
            img_channels=self.args.get("channels")
        )
        self.model = zari_model_wrapper.model
        if not self.model:
            raise ValueError("Aggregator failed to create a Keras model.")
        logger.info("Global Keras model created successfully.")

    def get_global_model_params(self):
        return self.model.get_weights()

    def set_global_model_params(self, weights):
        self.model.set_weights(weights)

    def get_global_keras_model(self):
        return self.model

    def aggregate(self, client_updates):
        if not client_updates:
            logger.warning("No client updates to aggregate. Skipping round.")
            return self.get_global_model_params()

        logger.info(f"Aggregating {len(client_updates)} client updates...")

        total_samples = 0
        base_weights = [np.zeros_like(p) for p in self.get_global_model_params()]

        for (num_samples, update_dict) in client_updates:
            client_weights = update_dict["weights"]
            total_samples += num_samples
            for i in range(len(base_weights)):
                base_weights[i] += (num_samples * client_weights[i])

        averaged_params = [layer_weights / total_samples for layer_weights in base_weights] if total_samples > 0 else self.get_global_model_params()
        
        self.set_global_model_params(averaged_params)
        logger.info("Aggregation complete. Global model updated.")

        self.evaluate()

        return self.get_global_model_params()

    def evaluate(self):
        if self.model is None or self.test_global is None:
            return

        logger.info("--- Starting global model evaluation... ---")

        test_images, test_labels = [], []
        for raw_sequences, labels in self.test_global:
            raw_sequences_np = raw_sequences.numpy()
            for seq in raw_sequences_np:
                pil_image = self.image_transformer.transform(seq)
                test_images.append(np.array(pil_image))
            test_labels.extend(labels.numpy())
            
        test_images = np.array(test_images).astype(np.float32) / 255.0
        test_labels = np.array(test_labels)

        if len(test_images) == 0:
            logger.warning("No data in test set to evaluate.")
            return
            
        results = self.model.evaluate(test_images, test_labels, verbose=0)
        loss, accuracy = results[0], results[1]
        
        logger.info(f"--- Global Model Evaluation Results ---")
        logger.info(f"    Loss: {loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"------------------------------------")