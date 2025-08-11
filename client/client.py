import logging
import numpy as np
import tensorflow as tf

from utils.img_transformer import CANImageTransformer
from utils.model import ZariTFLite 

logger = logging.getLogger("FL-Client")

class FedTFLiteClient:
    
    def __init__(self, client_id, fedml_args):
        self.client_id = client_id
        self.fedml_args = fedml_args
        self.model = None
        self.image_transformer = None
        logger.info(f"[Client {self.client_id}] Initializing...")

        if self.fedml_args.data.use_image_input:
            self.image_transformer = CANImageTransformer(img_size=self.fedml_args.data.img_size_default)
            logger.info(f"[Client {self.client_id}] CAN-to-Image Transformer initialized.")
        
        self.initialize_model()

    def set_id(self, client_id):
        self.client_id = client_id
        logger.info(f"[Client {self.client_id}] ID has been set by the framework.")

    def initialize_model(self):
        data_args = self.fedml_args.data
        zari_model_wrapper = ZariTFLite(
            input_size=data_args.input_size, num_classes=2, use_image_input=data_args.use_image_input,
            resource_profile="medium", img_height=data_args.img_size_default,
            img_width=data_args.img_size_default, img_channels=data_args.img_channels_default
        )
        self.model = zari_model_wrapper.model
        if self.model: logger.info(f"[Client {self.client_id}] Keras model created.")
        else: logger.error(f"[Client {self.client_id}] Keras model could not be created.")

    def get_weights(self):
        return self.model.get_weights() if self.model else None

    def set_weights(self, weights):
        if self.model and weights is not None: self.model.set_weights(weights)

    def train(self, train_data, device, args):
        if self.model is None: logger.error(f"[Client {self.client_id}] Model not initialized."); return

        logger.info(f"[Client {self.client_id}] Starting local training...")

        all_images, all_labels = [], []
        logger.info(f"[Client {self.client_id}] Transforming local raw data to images...")
        for raw_sequences, labels in train_data:
            raw_sequences_np = raw_sequences.numpy()
            for seq in raw_sequences_np:
                pil_image = self.image_transformer.transform(seq)
                all_images.append(np.array(pil_image))
            all_labels.extend(labels.numpy())

        if not all_images:
            logger.warning(f"[Client {self.client_id}] No data to train on. Skipping round.")
            return self.get_weights(), {"val_accuracy": 0, "val_loss": 0, "f1_score": 0} # Return dummy metrics
        
        final_images = np.array(all_images).astype(np.float32) / 255.0
        final_labels = np.array(all_labels)

        from sklearn.model_selection import train_test_split
        train_images, val_images, train_labels, val_labels = train_test_split(
            final_images, final_labels, test_size=0.1, random_state=42
        )
        
        logger.info(f"[Client {self.client_id}] Data ready. Starting model.fit() on {len(train_images)} samples...")
        with tf.device(device):
            self.model.fit(
                train_images, train_labels, 
                batch_size=args.train.batch_size,
                epochs=args.train.epochs, 
                validation_data=(val_images, val_labels),
                verbose=2,
                shuffle=True
            )
        
        logger.info(f"[Client {self.client_id}] Local training finished. Evaluating on local validation set...")
        val_loss, val_accuracy = self.model.evaluate(val_images, val_labels, verbose=0)
        logger.info(f"[Client {self.client_id}] Local validation accuracy: {val_accuracy*100:.2f}%")

        num_train_samples = len(train_images)

        model_update_dict = {
            "weights": self.get_weights(),
            "metrics": {
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
                "f1_score": 0.0 
            }
        }
        
        return num_train_samples, model_update_dict