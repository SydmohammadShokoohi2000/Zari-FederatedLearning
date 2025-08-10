import os
import sys
import argparse
import logging
import yaml
import fedml
from fedml.arguments import load_arguments

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FL-Main")

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return None

def run_server(args):
    if '.' not in sys.path:
        sys.path.insert(0, '.')

    from server.server import FedTFLiteServer

    config_path = args.config
    config = load_config(config_path)
    
    if config is None:
        logger.error("Failed to load configuration")
        return

    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(self, key, SimpleConfig(value))
                else:
                    setattr(self, key, value)

    fedml_config = SimpleConfig(config)

    if args.dataset_dir:
        if not hasattr(fedml_config, 'data'):
            fedml_config.data = SimpleConfig({})
        fedml_config.data.dataset_dir = args.dataset_dir
        
    if args.use_image_input is not None:
        if not hasattr(fedml_config, 'data'):
            fedml_config.data = SimpleConfig({})
        fedml_config.data.use_image_input = args.use_image_input
        
    if args.max_rounds:
        if not hasattr(fedml_config, 'train'):
            fedml_config.train = SimpleConfig({})
        fedml_config.train.max_rounds = args.max_rounds

    server = FedTFLiteServer(fedml_config)
    server.run()

def run_client(args):
    if '.' not in sys.path:
        sys.path.insert(0, '.')

    from client.client import FedTFLiteClient

    config_path = args.config
    config = load_config(config_path)
    
    if config is None:
        logger.error("Failed to load configuration")
        return

    fedml_args = argparse.Namespace()
    fedml_args.config = config_path
    fedml_args.training_type = "cross_silo"
    fedml_args.rank = int(args.client_id) if args.client_id.isdigit() else 1  
    fedml_args.run_id = "0"
    fedml_args.using_mlops = False  
    fedml_args.client_num_per_round = 2
    fedml_args.comm_round = 10

    fedml_args.is_mobile = True  
    fedml_args.device_type = args.device_type if args.device_type else "standard"
    fedml_args.enable_latency_measurement = True  

    if args.device_type == "resource_constrained":
        fedml_args.memory_optimizer = True
        fedml_args.batch_size = 16  

    fedml_config = fedml.init(fedml_args)

    if args.dataset_dir:
        fedml_config.data.dataset_dir = args.dataset_dir
    if args.use_image_input is not None:
        fedml_config.data.use_image_input = args.use_image_input

    client = FedTFLiteClient(args.client_id, fedml_config)
    client.run()

def run_simulation(args):
    from fedml.simulation import SimulatorMPI

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "client"))
    from server.server import FedTFLiteServer
    from client.client import FedTFLiteClient

    config_path = args.config
    config = load_config(config_path)
    
    if config is None:
        logger.error("Failed to load configuration")
        return

    fedml_config = load_arguments(config_path)

    if args.dataset_dir:
        fedml_config.data.dataset_dir = args.dataset_dir
    if args.use_image_input is not None:
        fedml_config.data.use_image_input = args.use_image_input
    if args.num_clients:
        fedml_config.federate.client_num_per_round = args.num_clients
    if args.max_rounds:
        fedml_config.train.max_rounds = args.max_rounds

    fedml.run_simulation(
        config=fedml_config,
        client_class=FedTFLiteClient,
        server_class=FedTFLiteServer,
        client_num=args.num_clients
    )

def generate_config(args):
    from config.fedml_config import generate_default_config

    generate_default_config(args.output)

def main():
    parser = argparse.ArgumentParser(description='FedML TensorFlow Lite IoV IDS')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    server_parser = subparsers.add_parser('server', help='Run server')
    server_parser.add_argument('--config', type=str, default='config/fedml_config.yaml',
                             help='Configuration file path')
    server_parser.add_argument('--dataset-dir', type=str,
                             help='Dataset directory')
    server_parser.add_argument('--use-image-input', action='store_true',
                             help='Use image-based input')
    server_parser.add_argument('--max-rounds', type=int,
                             help='Maximum number of rounds')

    client_parser = subparsers.add_parser('client', help='Run client')
    client_parser.add_argument('--client-id', type=str, required=True,
                             help='Client ID')
    client_parser.add_argument('--config', type=str, default='config/fedml_config.yaml',
                             help='Configuration file path')
    client_parser.add_argument('--dataset-dir', type=str,
                             help='Dataset directory')
    client_parser.add_argument('--use-image-input', action='store_true',
                             help='Use image-based input')
    client_parser.add_argument('--device-type', type=str, choices=['standard', 'resource_constrained'],
                             help='Device type')

    sim_parser = subparsers.add_parser('simulate', help='Run simulation')
    sim_parser.add_argument('--config', type=str, default='config/fedml_config.yaml',
                          help='Configuration file path')
    sim_parser.add_argument('--dataset-dir', type=str,
                          help='Dataset directory')
    sim_parser.add_argument('--use-image-input', action='store_true',
                          help='Use image-based input')
    sim_parser.add_argument('--num-clients', type=int, default=3,
                          help='Number of clients')
    sim_parser.add_argument('--num-processes', type=int, default=0,
                          help='Number of processes (0 for auto)')
    sim_parser.add_argument('--max-rounds', type=int,
                          help='Maximum number of rounds')

    config_parser = subparsers.add_parser('generate-config', help='Generate default configuration')
    config_parser.add_argument('--output', type=str, default='config/fedml_config.yaml',
                             help='Output configuration file path')
    
    args = parser.parse_args()
    
    if args.command == 'server':
        run_server(args)
    elif args.command == 'client':
        run_client(args)
    elif args.command == 'simulate':
        run_simulation(args)
    elif args.command == 'generate-config':
        generate_config(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()