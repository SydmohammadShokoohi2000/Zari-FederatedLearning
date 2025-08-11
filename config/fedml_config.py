import os
import yaml
import argparse

DEFAULT_CONFIG = {
    "environment_args": {
        "backend": "MQTT",
        "scenario": "cross_silo",
        "mqtt_host": "192.168.100.17", 
        "mqtt_port": 1883, 
    },
    "training_args": {
        "runner_type": "client",
        "training_type": "cross_silo",
        "federated_optimizer": "FedAvgCustom",
        "comm_round": 20,
        "client_num_per_round": 2,
        "epochs": 3, 
        "batch_size": 32, 
        "learning_rate": 0.001, 
        "weight_decay": 0.0001,
        "detection_threshold": 0.6, 
        "resource_weight_factor": 0.5, 
        "momentum_beta": 0.85, 
        "model_dir": "models_server_output" 
    },
    "data_args": {
        "dataset_name": "CarHackingIDS",
        "dataset_dir": "/home/mohammad/mobilenetv3.pytorch/9) Car-Hacking Dataset",
        "dataset_files": [
            "DoS_dataset.csv",
            "Fuzzy_dataset.csv",
            "gear_dataset.csv",
            "RPM_dataset.csv",
            "normal_run_data.txt"
        ],
        "use_image_input": False,
        "input_size": 13, 
        "window_size": 64,
        "img_size_default": 64,
        "img_channels_default": 1,
        "img_channels_high": 3
    },
    "client_args": {
        "stream_interval": 0.1,
        "attack_injection_rate": 0.01,
        "max_streaming_samples_test": 1000
    },
    "server_args": { 
        "min_clients_required_for_aggregation": 2
    },
    "common_args":{
        "using_mlops": False,
        "run_id": "iov_fl_run_default_001" 
    },
    "device_args":{
        "using_gpu": False,
        "gpu_id": None
    }
}

def generate_default_config(config_file_path="config/fedml_config.yaml"): 
    config_dir = os.path.dirname(config_file_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
        print(f"Created directory: {config_dir}")

    try:
        with open(config_file_path, 'w') as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)
        print(f"Default FedML configuration with new structure saved to: {config_file_path}")
        print("\nPLEASE REVIEW AND CUSTOMIZE THE GENERATED YAML FILE, especially:")
        print("  - 'environment_args.mqtt_host':")
        print("    - For SERVER: Set to '0.0.0.0' or its specific reachable IP.")
        print("    - For CLIENTs: Set to your Server VM's actual IP address.")
        print("  - 'environment_args.mqtt_port': Ensure it matches your MQTT broker's port (e.g., 1883).")
        print("  - 'data_args.dataset_dir': Ensure this path is correct on each client VM.")
        print(f"  - 'data_args.input_size': Currently {DEFAULT_CONFIG['data_args']['input_size']}. Verify this matches your feature vector size.")
    except Exception as e:
        print(f"Error saving default configuration to {config_file_path}: {e}")

if __name__ == "__main__":
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, '..'))

    parser = argparse.ArgumentParser(description='Generate Default FedML Configuration File with New Structure')
    parser.add_argument(
        '--output_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'config', 'fedml_config.yaml'), 
        help='Path to save the generated fedml_config.yaml file'
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory for output: {output_dir}")
         
    generate_default_config(args.output_path)