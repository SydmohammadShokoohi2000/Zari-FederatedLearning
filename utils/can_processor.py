import numpy as np
import pandas as pd
import re

class CANProcessor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.message_history = {}
        self.time_intervals = {}
    
    def extract_can_features(self, message):
        try:
            if isinstance(message, pd.DataFrame):
                message = message.iloc[0, 0]
            elif isinstance(message, pd.Series):
                message = message.iloc[0]
                
            features = []
            
            timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', message)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else 0
            features.append(timestamp % 1000)
            
            id_match = re.search(r'ID: ([0-9a-fA-F]+)', message)
            can_id = int(id_match.group(1), 16) if id_match else 0
            features.append(can_id / 2048.0)
            
            dlc_match = re.search(r'DLC: (\d+)', message)
            dlc = int(dlc_match.group(1)) if dlc_match else 0
            features.append(dlc / 8.0)
            
            data_bytes = []
            data_match = re.search(r'DLC: \d+\s+((?:[0-9a-fA-F]{2}\s*)+)', message)
            if data_match:
                data_str = data_match.group(1).strip()
                data_bytes = [int(b, 16) for b in data_str.split() if b]
                
                if data_bytes:
                    features.append(np.mean(data_bytes) / 255.0)
                    
                    counts = np.bincount(data_bytes, minlength=256)
                    probs = counts[counts > 0] / len(data_bytes)
                    entropy = -np.sum(probs * np.log2(probs)) / 8.0 if len(probs) > 0 else 0
                    features.append(entropy)
                    
                    features.append(np.std(data_bytes) / 255.0)
                    features.append(np.max(data_bytes) / 255.0)
                    features.append(np.min(data_bytes) / 255.0)
                else:
                    features.extend([0, 0, 0, 0, 0])
            else:
                features.extend([0, 0, 0, 0, 0])
                
            while len(features) < 8:
                features.append(0)
                
            return np.array(features)
                
        except Exception as e:
            print(f"[CAN] Error extracting CAN features: {str(e)}")
            return np.zeros(8)
    
    def process_message(self, message):
        try:
            features = self.extract_can_features(message)
            
            if isinstance(message, pd.DataFrame):
                message_str = message.iloc[0, 0]
            elif isinstance(message, pd.Series):
                message_str = message.iloc[0]
            else:
                message_str = message
                
            id_match = re.search(r'ID: ([0-9a-fA-F]+)', message_str)
            if not id_match:
                return features, np.zeros(5)
                
            can_id = id_match.group(1)
            timestamp_match = re.search(r'Timestamp: (\d+\.\d+)', message_str)
            timestamp = float(timestamp_match.group(1)) if timestamp_match else 0
            
            if can_id in self.message_history:
                last_time = self.message_history[can_id][-1]['timestamp']
                interval = timestamp - last_time
                
                if can_id not in self.time_intervals:
                    self.time_intervals[can_id] = []
                
                if len(self.time_intervals[can_id]) >= self.window_size:
                    self.time_intervals[can_id].pop(0)
                
                self.time_intervals[can_id].append(interval)
            
            data_match = re.search(r'DLC: \d+\s+((?:[0-9a-fA-F]{2}\s*)+)', message_str)
            data_bytes = []
            if data_match:
                data_str = data_match.group(1).strip()
                data_bytes = [int(b, 16) for b in data_str.split() if b]
            
            if can_id not in self.message_history:
                self.message_history[can_id] = []
                
            if len(self.message_history[can_id]) >= self.window_size:
                self.message_history[can_id].pop(0)
                
            self.message_history[can_id].append({
                'timestamp': timestamp,
                'data': data_bytes
            })
            
            temporal_features = self._extract_temporal_features(can_id)
            
            return features, temporal_features
            
        except Exception as e:
            print(f"[CAN] Error processing message: {str(e)}")
            return np.zeros(8), np.zeros(5)
    
    def _extract_temporal_features(self, can_id):
        features = []
        
        if can_id not in self.message_history or len(self.message_history[can_id]) < 2:
            return np.zeros(5)
            
        history = self.message_history[can_id]
        
        if can_id in self.time_intervals and len(self.time_intervals[can_id]) > 1:
            intervals = self.time_intervals[can_id]
            features.append(np.mean(intervals))
            features.append(np.std(intervals))
            features.append(np.max(intervals) / (np.mean(intervals) + 1e-8))
        else:
            features.extend([0, 0, 0])
            
        payload_features = []
        if len(history) > 1 and all(len(msg['data']) > 0 for msg in history):
            byte_positions = min(len(msg['data']) for msg in history)
            byte_variances = []
            
            for pos in range(min(byte_positions, 2)):
                values = [msg['data'][pos] for msg in history]
                byte_variances.append(np.var(values))
                
            if byte_variances:
                payload_features.append(np.mean(byte_variances) / 255.0)
                payload_features.append(1.0 if np.max(byte_variances) == 0 else 0.0)
        
        features.extend(payload_features[:2] if payload_features else [0, 0])
        
        while len(features) < 5:
            features.append(0)
            
        return np.array(features)
    
    def reset(self):
        self.message_history = {}
        self.time_intervals = {}
    
    def convert_to_tflite_input(self, messages, window_size=None):
        if window_size is None:
            window_size = self.window_size
        
        if isinstance(messages, pd.DataFrame):
            messages = messages.iloc[:window_size]
        elif isinstance(messages, list) and len(messages) > window_size:
            messages = messages[:window_size]
        
        features_list = []
        
        for msg in messages:
            if msg is not None:
                basic_features, temporal_features = self.process_message(msg)
                features = np.concatenate([basic_features, temporal_features])
                features_list.append(features)
        
        if not features_list:
            return np.zeros((1, 13), dtype=np.float32)
        
        combined_features = np.array(features_list)
        
        if combined_features.shape[0] < window_size:
            padding = np.zeros((window_size - combined_features.shape[0], combined_features.shape[1]))
            combined_features = np.vstack((combined_features, padding))
        
        return combined_features.reshape(1, -1).astype(np.float32)
    
    def analyze_can_traffic(self, messages, detection_threshold=0.7):
        if len(messages) < self.window_size:
            return {
                "status": "insufficient_data",
                "details": f"Need at least {self.window_size} messages, got {len(messages)}"
            }
        
        metrics = {}
        
        try:
            interval_analysis = self._analyze_intervals(messages)
            content_analysis = self._analyze_content(messages)
            
            anomaly_score = (interval_analysis["anomaly_score"] + content_analysis["anomaly_score"]) / 2
            
            is_anomalous = anomaly_score > detection_threshold
            
            metrics = {
                "anomaly_score": anomaly_score,
                "is_anomalous": is_anomalous,
                "interval_metrics": interval_analysis,
                "content_metrics": content_analysis
            }
        except Exception as e:
            metrics = {
                "status": "error",
                "details": str(e)
            }
        
        return metrics
    
    def _analyze_intervals(self, messages):
        id_intervals = {}
        
        for i in range(1, len(messages)):
            if isinstance(messages[i-1], pd.Series) and isinstance(messages[i], pd.Series):
                msg1 = messages[i-1].iloc[0] if isinstance(messages[i-1].iloc[0], str) else str(messages[i-1].iloc[0])
                msg2 = messages[i].iloc[0] if isinstance(messages[i].iloc[0], str) else str(messages[i].iloc[0])
                
                id1_match = re.search(r'ID: ([0-9a-fA-F]+)', msg1)
                id2_match = re.search(r'ID: ([0-9a-fA-F]+)', msg2)
                
                ts1_match = re.search(r'Timestamp: (\d+\.\d+)', msg1)
                ts2_match = re.search(r'Timestamp: (\d+\.\d+)', msg2)
                
                if id1_match and id2_match and ts1_match and ts2_match:
                    can_id = id1_match.group(1)
                    
                    if can_id == id2_match.group(1):
                        t1 = float(ts1_match.group(1))
                        t2 = float(ts2_match.group(1))
                        interval = t2 - t1
                        
                        if can_id not in id_intervals:
                            id_intervals[can_id] = []
                        
                        id_intervals[can_id].append(interval)
        
        anomaly_score = 0.0
        interval_std_ratio = 0.0
        irregular_ids = 0
        
        for can_id, intervals in id_intervals.items():
            if len(intervals) > 5:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                variation = std_interval / mean_interval if mean_interval > 0 else 0
                
                if variation > 0.5:
                    irregular_ids += 1
                    anomaly_score += min(1.0, variation)
        
        if id_intervals:
            anomaly_score = anomaly_score / len(id_intervals)
        
        return {
            "anomaly_score": anomaly_score,
            "irregular_ids": irregular_ids,
            "total_ids": len(id_intervals)
        }
    
    def _analyze_content(self, messages):
        id_contents = {}
        
        for msg in messages:
            if isinstance(msg, pd.Series):
                msg_str = msg.iloc[0] if isinstance(msg.iloc[0], str) else str(msg.iloc[0])
                
                id_match = re.search(r'ID: ([0-9a-fA-F]+)', msg_str)
                data_match = re.search(r'DLC: \d+\s+((?:[0-9a-fA-F]{2}\s*)+)', msg_str)
                
                if id_match and data_match:
                    can_id = id_match.group(1)
                    data_str = data_match.group(1).strip()
                    data_bytes = [int(b, 16) for b in data_str.split() if b]
                    
                    if can_id not in id_contents:
                        id_contents[can_id] = []
                    
                    id_contents[can_id].append(data_bytes)
        
        anomaly_score = 0.0
        unusual_patterns = 0
        
        for can_id, contents_list in id_contents.items():
            if len(contents_list) > 5:
                content_array = np.array([c for c in contents_list if len(c) > 0])
                
                if content_array.size > 0:
                    value_range = np.max(content_array) - np.min(content_array)
                    
                    if value_range > 200:
                        unusual_patterns += 1
                        anomaly_score += min(1.0, value_range / 255.0)
        
        if id_contents:
            anomaly_score = anomaly_score / len(id_contents)
        
        return {
            "anomaly_score": anomaly_score,
            "unusual_patterns": unusual_patterns,
            "total_ids": len(id_contents)
        }