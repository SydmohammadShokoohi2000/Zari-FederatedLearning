import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import fedml
from fedml.simulation import SimulatorMPI
from fedml.cross_silo import Client

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from client.model import ZariTFLite
from utils.can_processor import CANProcessor
from utils.img_transformer import CANImageTransformer, StreamingCANDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FL-Client")

class FedTFLiteClient(Client):
    def __init__(self, client_id, args):
        super().__init__(client_id, args)
        
        self.client_id = client_id
        self.args = args
        self.device_type = args.device_type if hasattr(args, "device_type") else "standard"
        self.dataset_dir = args.dataset_dir if hasattr(args, "dataset_dir") else "/data"
        self.use_image_input = args.use_image_input if hasattr(args, "use_image_input") else False
        
        self.input_size = args.input_size if hasattr(args, "input_size") else 8
        self.learning_rate = args.learning_rate if hasattr(args, "learning_rate") else 0.0002
        self.local_epochs = args.local_epochs if hasattr(args, "local_epochs") else 5
        self.batch_size = args.batch_size if hasattr(args, "batch_size") else 64
        
        self.streaming_active = False
        self.detection_threshold = args.detection_threshold if hasattr(args, "detection_threshold") else 0.7
        self.stream_interval = args.stream_interval if hasattr(args, "stream_interval") else 0.1
        self.attack_injection_rate = args.attack_injection_rate if hasattr(args, "attack_injection_rate") else 0.01
        
        self.detection_stats = {
            "total_samples": 0, 
            "attack_detected": 0,
            "normal_detected": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "detection_times": []
        }
        
        self._init_model()
        
        self.can_processor = CANProcessor(window_size=20)
        
        if self.use_image_input:
            self.streaming_dataset = StreamingCANDataset(
                window_size=64,
                img_size=224
            )
        
        logger.info(f"Client {client_id} initialized with {self.device_type} device type")
        logger.info(f"Using {'image-based' if self.use_image_input else 'feature-based'} input")
    
    def _init_model(self):
        self.model = ZariTFLite(
            input_size=self.input_size,
            num_classes=2,
            use_image_input=self.use_image_input,
            device_type=self.device_type
        )

        if self.device_type == "resource_constrained":
            if not self.use_image_input:
                self.input_size = min(self.input_size, 8)

            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    logger.warning(f"Memory growth setting failed: {str(e)}")

            tf.config.threading.set_inter_op_parallelism_threads(2)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        
        logger.info(f"Model initialized with input size {self.input_size} for {self.device_type} device")

    def load_data(self, dataset_files=None):
        if dataset_files is None:
            dataset_files = [
                "DoS_dataset.csv",
                "Fuzzy_dataset.csv",
                "gear_dataset.csv",
                "RPM_dataset.csv",
                "normal_run_data.txt"
            ]
        
        logger.info(f"Loading data from {len(dataset_files)} files")
        
        all_X = []
        all_y = []
        
        for file in dataset_files:
            try:
                file_path = os.path.join(self.dataset_dir, file)
                
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                chunk_size = 50000
                df_chunks = pd.read_csv(file_path, chunksize=chunk_size)
                
                for df in df_chunks:
                    if 'Label' not in df.columns:
                        df['Label'] = 0 if 'normal' in file.lower() else 1
                    
                    X, y = self._preprocess_data(df)
                    
                    if X is not None and len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)
                
                logger.info(f"Loaded data from {file}")
                
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
        
        if not all_X:
            logger.error("No data loaded")
            return None, None, None, None
        
        X = np.vstack(all_X)
        y = np.concatenate(all_y)
        
        X_train, X_val, y_train, y_val = self._train_test_split(X, y)
        
        logger.info(f"Data loaded: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
        
        return X_train, y_train, X_val, y_val
    
    def _preprocess_data(self, df):
        try:
            if 'Timestamp' in df.columns:
                df = df.drop(['Timestamp'], axis=1)
            
            if 'CAN_ID' in df.columns:
                if df['CAN_ID'].dtype == 'object':
                    df['CAN_ID'] = df['CAN_ID'].apply(lambda x: int(x, 16) if isinstance(x, str) else x)
                
                agg_cols = [col for col in df.columns if col not in ['CAN_ID', 'Label']]
                
                if agg_cols:
                    can_features = df.groupby('CAN_ID')[agg_cols].agg(['mean', 'std', 'min', 'max', 'median'])
                    can_features = can_features.reset_index()
                    
                    can_features.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in can_features.columns.values]
                    
                    label_df = df.groupby('CAN_ID')['Label'].max().reset_index()
                    
                    processed_df = pd.merge(can_features, label_df, on='CAN_ID', how='left')
                else:
                    processed_df = df
            else:
                processed_df = df
            
            processed_df = processed_df.fillna(0)
            
            feature_columns = [col for col in processed_df.columns if col != 'Label']
            X = processed_df[feature_columns].values.astype(np.float32)
            y = processed_df['Label'].values.astype(np.int32)
            
            if self.use_image_input:
                X_images, y_images = self._convert_to_image_dataset(processed_df)
                if X_images is not None:
                    return X_images, y_images
            
            if X.shape[1] != self.input_size:
                if X.shape[1] < self.input_size:
                    padding = np.zeros((X.shape[0], self.input_size - X.shape[1]))
                    X = np.hstack((X, padding))
                else:
                    X = X[:, :self.input_size]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None, None
    
    def _convert_to_image_dataset(self, df, window_size=64, img_size=224):
        try:
            transformer = CANImageTransformer(window_size=window_size, img_size=img_size)
            
            X_images = []
            y_labels = []
            
            for i in range(0, len(df) - window_size + 1, window_size // 2):
                window = df.iloc[i:i+window_size]
                is_attack = window['Label'].max() > 0  
                
                img = transformer.transform(window, is_attack)
                img_array = np.array(img).astype(np.float32) / 255.0
                
                if img_array.shape[2] == 3:  # Ensure we have RGB channels
                    X_images.append(img_array)
                    y_labels.append(1 if is_attack else 0)
            
            if X_images:
                X_images_array = np.array(X_images)
                y_labels_array = np.array(y_labels)
                
                return X_images_array, y_labels_array
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Error converting to images: {str(e)}")
            return None, None
    
    def _train_test_split(self, X, y, test_size=0.2, random_state=42):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        indices = np.arange(len(y))
        indices_train, indices_val = [], []
        
        for c in unique_classes:
            c_indices = indices[y == c]
            np.random.shuffle(c_indices)
            
            split_idx = int(len(c_indices) * (1 - test_size))
            indices_train.extend(c_indices[:split_idx])
            indices_val.extend(c_indices[split_idx:])
        
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_val)
        
        X_train, y_train = X[indices_train], y[indices_train]
        X_val, y_val = X[indices_val], y[indices_val]
        
        return X_train, X_val, y_train, y_val
    
    def on_receive_model(self, model_weights):
        logger.info("Received global model from server")
        
        try:
            self.model.load_tflite_model(model_content=model_weights)
            logger.info("Global model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading global model: {str(e)}")
    
    def train(self, X_train, y_train, X_val, y_val):
        logger.info("Starting local training")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(self.batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = self.model.model(x, training=True)
                loss = loss_fn(y, predictions)
            
            gradients = tape.gradient(loss, self.model.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.model.trainable_variables))
            
            train_loss(loss)
            train_accuracy(y, predictions)
        
        @tf.function
        def test_step(x, y):
            predictions = self.model.model(x, training=False)
            loss = loss_fn(y, predictions)
            
            val_loss(loss)
            val_accuracy(y, predictions)
        
        for epoch in range(self.local_epochs):
            train_loss.reset_states()
            train_accuracy.reset_states()
            val_loss.reset_states()
            val_accuracy.reset_states()
            
            for x_batch, y_batch in train_dataset:
                train_step(x_batch, y_batch)
            
            for x_batch, y_batch in val_dataset:
                test_step(x_batch, y_batch)
            
            logger.info(
                f"Epoch {epoch+1}/{self.local_epochs}, "
                f"Loss: {train_loss.result():.4f}, "
                f"Accuracy: {train_accuracy.result()*100:.2f}%, "
                f"Val Loss: {val_loss.result():.4f}, "
                f"Val Accuracy: {val_accuracy.result()*100:.2f}%"
            )
        
        self.model.convert_to_tflite(quantize=True)
        
        y_pred = []
        y_true = []
        
        for x_batch, y_batch in val_dataset:
            predictions = self.model.model(x_batch, training=False)
            pred_labels = tf.argmax(predictions, axis=1).numpy()
            y_pred.extend(pred_labels)
            y_true.extend(y_batch.numpy())
        
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        accuracy = np.mean(y_pred == y_true)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "val_loss": val_loss.result().numpy(),
            "val_accuracy": accuracy * 100,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        logger.info(f"Training complete. Metrics: {metrics}")
        
        return metrics
    
    def run(self):
        logger.info(f"Starting client {self.client_id}")
        
        X_train, y_train, X_val, y_val = self.load_data()
        
        if X_train is None:
            logger.error("Failed to load data. Exiting.")
            return
        
        fedml.run_simulation(
            self.train,
            server_class=None,
            client_class=FedTFLiteClient,
            args=self.args
        )
    
    def inject_attack(self, sample):
        attack_sample = sample.copy()
        
        try:
            if isinstance(sample, pd.Series):
                if isinstance(sample.iloc[0], str) and "ID:" in sample.iloc[0]:
                    message = sample.iloc[0]
                    
                    attack_type = np.random.choice(['spoof', 'spoof', 'dos', 'replay'])
                    
                    if "DLC:" in message:
                        parts = message.split("DLC:")
                        header = parts[0] + "DLC:"
                        data_parts = parts[1].split()
                        
                        if len(data_parts) > 1:
                            dlc = int(data_parts[0])
                            
                            if attack_type == 'spoof':
                                if dlc > 2 and len(data_parts) > 2:
                                    data_parts[1] = "FF"
                                    if len(data_parts) > 2:
                                        data_parts[2] = "FF"
                            
                            elif attack_type == 'replay':
                                import re
                                timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', message)
                                if timestamp_match:
                                    old_ts = timestamp_match.group(0)
                                    new_ts = f"Timestamp: {float(timestamp_match.group(1)) - 0.5:.6f}"
                                    header = header.replace(old_ts, new_ts)
                            
                            elif attack_type == 'dos':
                                for i in range(1, min(dlc + 1, len(data_parts))):
                                    data_parts[i] = "FF"
                            
                            new_message = header + " " + " ".join(data_parts)
                            attack_sample.iloc[0] = new_message
                
            else:
                numeric_cols = attack_sample.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    col = np.random.choice(numeric_cols)
                    attack_sample[col] = attack_sample[col] * 10
        
        except Exception as e:
            logger.error(f"Error injecting attack: {str(e)}")
        
        return attack_sample
    
    def detect_anomaly(self, features):
        start_time = time.time()
        
        try:
            if isinstance(features, np.ndarray):
                if features.dtype != np.float32:
                    features = features.astype(np.float32)
            
            if self.use_image_input and len(features.shape) == 4:
                prediction, confidence = self.model.detect_anomaly(features)
            else:
                if len(features.shape) == 1:
                    features = np.reshape(features, (1, -1))
                
                prediction, confidence = self.model.detect_anomaly(features)
            
            detection_time = time.time() - start_time
            self.detection_stats["detection_times"].append(detection_time)
            
            self.detection_stats["total_samples"] += 1
            
            logger.debug(
                f"Detection result: {'Attack' if prediction == 1 else 'Normal'}, "
                f"Confidence: {confidence:.4f}, "
                f"Time: {detection_time*1000:.2f} ms"
            )
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return 0, 0.0
    
    def start_streaming_detection(self, normal_buffer):
        self.streaming_active = True
        sample_index = 0
        buffer_size = len(normal_buffer)
        
        logger.info(f"Starting streaming detection with {buffer_size} samples in buffer")
        
        while self.streaming_active:
            try:
                start_time = time.time()
                
                sample = normal_buffer.iloc[sample_index % buffer_size].copy()
                sample_index += 1
                
                is_attack = False
                if np.random.random() < self.attack_injection_rate:
                    is_attack = True
                    sample = self.inject_attack(sample)
                
                if self.use_image_input:
                    self.streaming_dataset.add_message(sample)
                    features = self.streaming_dataset.get_current_tensor()
                    
                    if features is None:
                        continue
                else:
                    basic_features, temporal_features = self.can_processor.process_message(sample)
                    features = np.concatenate([basic_features, temporal_features])
                
                prediction, confidence = self.detect_anomaly(features)
                
                if prediction == 1:
                    self.detection_stats["attack_detected"] += 1
                    if not is_attack:
                        self.detection_stats["false_positives"] += 1
                else:
                    self.detection_stats["normal_detected"] += 1
                    if is_attack:
                        self.detection_stats["false_negatives"] += 1
                
                if self.detection_stats["total_samples"] % 1000 == 0:
                    self._report_detection_stats()
                
                elapsed = time.time() - start_time
                if elapsed < self.stream_interval:
                    time.sleep(self.stream_interval - elapsed)
                    
            except Exception as e:
                logger.error(f"Error in streaming detection: {str(e)}")
                time.sleep(0.1)
    
    def _report_detection_stats(self):
        stats = self.detection_stats
        total = stats["total_samples"]
        attacks = stats["attack_detected"]
        normals = stats["normal_detected"]
        fps = stats["false_positives"]
        fns = stats["false_negatives"]
        
        detection_rate = attacks / (attacks + fns) if (attacks + fns) > 0 else 0
        false_positive_rate = fps / (fps + normals - fns) if (fps + normals - fns) > 0 else 0
        
        avg_time = np.mean(stats["detection_times"]) if stats["detection_times"] else 0
        
        logger.info(
            f"Detection Stats: Samples={total}, "
            f"Attacks={attacks}, Normal={normals}, "
            f"FP={fps}, FN={fns}, "
            f"Detection Rate={detection_rate:.4f}, "
            f"FP Rate={false_positive_rate:.4f}, "
            f"Avg Time={avg_time*1000:.2f} ms"
        )
        
        stats["detection_times"] = []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='FedML TFLite Client')
    parser.add_argument('--client_id', type=str, default='client_1', help='Client ID')
    parser.add_argument('--device_type', type=str, default='standard', help='Device type (standard or resource_constrained)')
    parser.add_argument('--dataset_dir', type=str, default='/data', help='Dataset directory')
    parser.add_argument('--use_image_input', action='store_true', help='Use image-based input')
    parser.add_argument('--input_size', type=int, default=8, help='Input feature size')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--detection_threshold', type=float, default=0.7, help='Detection threshold')
    parser.add_argument('--stream_interval', type=float, default=0.1, help='Streaming interval in seconds')
    parser.add_argument('--attack_injection_rate', type=float, default=0.01, help='Attack injection rate')
    parser.add_argument('--config', type=str, default='../config/fedml_config.yaml', help='FedML config file')
    
    args = parser.parse_args()
    
    fedml_config = fedml.integration.load_arguments(args.config)
    
    for key, value in vars(args).items():
        setattr(fedml_config, key, value)
    
    client = FedTFLiteClient(args.client_id, fedml_config)
    client.run()

if __name__ == "__main__":
    main()