import os
import sys
import logging
import time
import yaml
import numpy as np
import tensorflow as tf
from .aggregator import TFLiteAggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FL-Server")

class FedTFLiteServer:
    def __init__(self, args):
        self.args = args
        self.device_type = "server"
        
        self.use_image_input = getattr(args.data, "use_image_input", False) if hasattr(args, "data") else False
        self.input_size = getattr(args.data, "input_size", 8) if hasattr(args, "data") else 8
        
        self.max_rounds = getattr(args.train, "max_rounds", 10) if hasattr(args, "train") else 10
        self.min_clients_required = getattr(args.train, "min_clients_required", 2) if hasattr(args, "train") else 2
        self.resource_weight_factor = getattr(args.train, "resource_weight_factor", 0.5) if hasattr(args, "train") else 0.5
        self.momentum_beta = getattr(args.train, "momentum_beta", 0.85) if hasattr(args, "train") else 0.85
        
        self.aggregator = TFLiteAggregator(
            input_size=self.input_size,
            use_image_input=self.use_image_input,
            resource_weight_factor=self.resource_weight_factor,
            momentum_beta=self.momentum_beta
        )
        
        self.aggregator.initialize_global_model(self.input_size)
        
        self.client_profiles = {}
        self.round_history = {}
        self.current_round = 0
        
        self.model_dir = getattr(args.train, "model_dir", "models") if hasattr(args, "train") else "models"
        os.makedirs(self.model_dir, exist_ok=True)

        self.client_list = []
        
        logger.info(f"Server initialized with max_rounds={self.max_rounds}")
        logger.info(f"Using {'image-based' if self.use_image_input else 'feature-based'} input with size {self.input_size}")
    
    def get_model_params(self):
        if self.current_round == 0:
            return self.aggregator.get_global_model()
        else:
            return self.get_global_model_params()
    
    def get_global_model_params(self):
        return self.aggregator.get_global_model()
    
    def get_lightweight_model_params(self):
        return self.aggregator.get_lightweight_model()
    
    def run(self):
        logger.info("Starting server in standalone mode for resource-constrained IoV environments")
        logger.info("A model has been initialized. To distribute it to clients, they need to connect to this server.")
        logger.info(f"Server is running and waiting for clients. Initial model is ready in {self.model_dir}")

        self._save_model("initial_model.tflite")

        logger.info("Server is ready. Run client instances to connect to this server.")
        logger.info("Press Ctrl+C to stop the server.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server stopped by user.")
    
    def register_client(self, client_id, profile=None):
        if profile is None:
            profile = {}
        
        self.client_profiles[client_id] = profile

        if client_id not in self.client_list:
            self.client_list.append(client_id)
            
        logger.info(f"Client {client_id} registered with profile: {profile}")
    
    def unregister_client(self, client_id):
        if client_id in self.client_profiles:
            del self.client_profiles[client_id]

            if client_id in self.client_list:
                self.client_list.remove(client_id)
                
            logger.info(f"Client {client_id} unregistered")
    
    def _save_model(self, filename):
        filepath = os.path.join(self.model_dir, filename)
        
        try:
            tflite_model = self.aggregator.get_global_model()
            with open(filepath, 'wb') as f:
                f.write(tflite_model)
            
            logger.info(f"Model saved to {filepath}")

            lightweight_filepath = os.path.join(self.model_dir, f"lightweight_{filename}")
            lightweight_model = self.aggregator.get_lightweight_model()
            with open(lightweight_filepath, 'wb') as f:
                f.write(lightweight_model)
                
            logger.info(f"Lightweight model saved to {lightweight_filepath}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def _save_final_model(self):
        self._save_model("final_model.tflite")

    def send_message_model_to_client(self, client_id, model_params):
        logger.info(f"Sending model to client {client_id}")
    
    def receive_model_from_client(self, client_id):
        logger.info(f"Received model from client {client_id}")
        return None  