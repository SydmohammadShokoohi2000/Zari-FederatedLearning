import sys
import os
import pandas as pd
import numpy as np
import logging
from PIL import Image
import re
import h5py

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.can_processor import CANProcessor
from utils.img_transformer import CANImageTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PREPROCESSOR")



def parse_can_value(value_str):
    value_str = str(value_str).strip().lower()
    if not value_str: return 0
    if value_str.startswith("0x") or any(c in "abcdef" for c in value_str):
        try: return int(value_str, 16)
        except ValueError:
            try: return int(value_str) 
            except ValueError: return 0
    else:
        try: return int(value_str)
        except ValueError: return 0

def extract_9_features_from_csv_row(row_series):
    features = np.zeros(9, dtype=np.int32)
    try:
        features[0] = parse_can_value(row_series.iloc[1])
        for i in range(8):
            col_index = 3 + i
            if col_index < len(row_series):
                features[i+1] = parse_can_value(row_series.iloc[col_index])
    except Exception: pass 
    return features.tolist()

def extract_9_features_from_txt_line(line_str):
    features = np.zeros(9, dtype=np.int32)
    try:
        id_match = re.search(r'ID: ([0-9a-fA-F]+)', line_str)
        if id_match: features[0] = int(id_match.group(1), 16)
        data_match = re.search(r'DLC: \d+\s+((?:[0-9a-fA-F]{2}\s*)+)', line_str)
        if data_match:
            data_str = data_match.group(1).strip()
            data_bytes = [int(b, 16) for b in data_str.split() if b]
            for i in range(min(len(data_bytes), 8)):
                features[i+1] = data_bytes[i]
    except Exception: pass 
    return features.tolist()

def load_can_data_as_9_features(filepath, is_attack_file=True):
    feature_vectors, derived_labels = [], []
    file_default_label = 1 if is_attack_file else 0
    logger.info(f"Loading from {filepath}...")
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath, header=None, dtype=str, on_bad_lines='skip', low_memory=False)
            for index, row in df.iterrows():
                if len(row) < 11: continue 
                features = extract_9_features_from_csv_row(row)
                feature_vectors.append(features)
                label_to_assign = file_default_label
                label_char = str(row.iloc[11]).strip().upper()
                if label_char == 'R': label_to_assign = 0
                elif label_char == 'T': label_to_assign = 1 
                derived_labels.append(label_to_assign)
        elif filepath.endswith(".txt"):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line: 
                        features = extract_9_features_from_txt_line(line)
                        feature_vectors.append(features)
                        derived_labels.append(file_default_label)
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}", exc_info=True)
    logger.info(f"Loaded {len(feature_vectors)} feature sets from {filepath}")
    return feature_vectors, derived_labels

def create_sequences_from_features(all_feature_vectors, all_labels, window_size):
    sequences, sequence_labels = [], []
    if len(all_feature_vectors) < window_size:
        logger.warning(f"Not enough feature_vectors ({len(all_feature_vectors)}) to form a window of size {window_size}.")
        return sequences, sequence_labels 
        
    for i in range(len(all_feature_vectors) - window_size + 1):
        sequences.append(all_feature_vectors[i : i + window_size])

        labels_in_window = all_labels[i : i + window_size]
        if 1 in labels_in_window: 
            sequence_labels.append(1) 
        else:
            sequence_labels.append(0) 
            
    logger.info(f"Created {len(sequences)} sequences, window_size {window_size}.")
    return sequences, sequence_labels

def main():
    logger.info("Starting memory-efficient data pre-processing.")
    DATASET_DIR = "/home/mohammad/mobilenetv3.pytorch/9) Car-Hacking Dataset"
    DATASET_FILES = ["DoS_dataset.csv", "Fuzzy_dataset.csv", "gear_dataset.csv", "RPM_dataset.csv", "normal_run_data.txt"]
    WINDOW_SIZE = 27
    IMG_SIZE = 64
    CHANNELS = 3
    OUTPUT_DIR = "/home/mohammad/FinalProject/data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Loading raw CAN data...")
    all_feature_vectors, all_derived_labels = [], []
    for filename in DATASET_FILES:
        filepath = os.path.join(DATASET_DIR, filename)
        is_attack = not ("normal" in filename.lower())
        features, labels = load_can_data_as_9_features(filepath, is_attack_file=is_attack)
        all_feature_vectors.extend(features); all_derived_labels.extend(labels)

    logger.info(f"Originally loaded {len(all_feature_vectors)} total feature sets.")
    SUBSET_SIZE = 400000
    if len(all_feature_vectors) > SUBSET_SIZE:
        all_feature_vectors = all_feature_vectors[:SUBSET_SIZE]
        all_derived_labels = all_derived_labels[:SUBSET_SIZE]
        logger.info(f"TRUNCATING to a subset of {len(all_feature_vectors)} for processing.")

    logger.info("Creating sequences from features...")
    sequences, sequence_labels = create_sequences_from_features(all_feature_vectors, all_derived_labels, WINDOW_SIZE)
    if not sequences:
        logger.error("No sequences were created. Exiting."); return
    
    num_sequences = len(sequences)
    logger.info(f"A total of {num_sequences} sequences will be processed and saved.")

    logger.info("Creating HDF5 file for output...")
    output_hdf5_path = os.path.join(OUTPUT_DIR, "processed_data.h5")
    
    with h5py.File(output_hdf5_path, 'w') as hf:
        dset_samples = hf.create_dataset("samples", shape=(num_sequences, IMG_SIZE, IMG_SIZE, CHANNELS), dtype='uint8')
        dset_labels = hf.create_dataset("labels", shape=(num_sequences,), dtype='int32')
        
        logger.info("Transforming sequences and saving directly to HDF5 file...")
        transformer = CANImageTransformer(img_size=IMG_SIZE)
        
        for i, seq in enumerate(sequences):
            label = sequence_labels[i]
            
            pil_image = transformer.transform(seq)
            img_array = np.array(pil_image) 
            
            dset_samples[i] = img_array
            dset_labels[i] = label

            if (i + 1) % 50000 == 0:
                logger.info(f"Processed {i + 1} / {num_sequences} images...")

    logger.info(f"--- PRE-PROCESSING COMPLETE ---")
    logger.info(f"Successfully saved all {num_sequences} processed samples and labels to '{output_hdf5_path}'.")

if __name__ == "__main__":
    main()