import sys
import os
import argparse
import logging
import yaml
import fedml
import tensorflow as tf
import time

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from client.client import FedTFLiteClient
    from server.aggregator import TFLiteAggregator
    from utils.data_loader import load_and_prepare_car_hacking_data_federated
    from config.fedml_config import generate_default_config as generate_yaml_config_script
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import custom modules: {e}")
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FL-EntryPoint")

def load_yaml_config_file(config_path):
    try:
        with open(os.path.abspath(config_path), 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML {config_path}: {e}", exc_info=True)
        return None

def run_server(cli_args_obj):
    logger.info("Starting FedML server...")
    yaml_config = load_yaml_config_file(cli_args_obj.config)
    if not yaml_config: return

    try:
        training_args_yaml = yaml_config.get('training_args', {})
        env_args_yaml = yaml_config.get('environment_args', {})
        common_args_yaml = yaml_config.get('common_args', {})
        
        fedml_init_params = argparse.Namespace(
            config_file=cli_args_obj.config, rank=int(cli_args_obj.rank), role='server',
            training_type=training_args_yaml.get('training_type', 'cross_silo'),
            backend=env_args_yaml.get('backend', 'MQTT'),
            scenario=env_args_yaml.get('scenario', 'horizontal'),
            federated_optimizer=training_args_yaml.get('federated_optimizer', 'FedAvg'),
            client_num_in_total=training_args_yaml.get('client_num_in_total', 2),
            client_num_per_round=training_args_yaml.get('client_num_per_round', 2),
            comm_round=training_args_yaml.get('comm_round', 20),
            epochs=training_args_yaml.get('epochs', 5),
            batch_size=training_args_yaml.get('batch_size', 64),
            learning_rate=training_args_yaml.get('learning_rate', 0.001),
            using_mlops=common_args_yaml.get('using_mlops', False),
            run_id=common_args_yaml.get('run_id', str(int(time.time()))),
        )

        args = fedml.init(args=fedml_init_params)
        if not args: logger.error("Server: fedml.init() failed."); return

        client_id_list = list(range(1, args.client_num_in_total + 1))
        dataset_info_list = load_and_prepare_car_hacking_data_federated(
            processed_data_dir=os.path.join(PROJECT_ROOT, "data", "processed"),
            num_clients=args.client_num_in_total,
            client_ids=client_id_list,
            batch_size=args.batch_size
        )
        test_data_loader = dataset_info_list[3]
        num_actual_classes = dataset_info_list[7]

        aggregator = TFLiteAggregator(
            args=args,
            test_global=test_data_loader,
            num_classes=num_actual_classes
        )
        model = aggregator.get_global_keras_model()

        server_runner = fedml.FedMLRunner(
            args=args,
            device=None, 
            dataset=dataset_info_list,
            model=model,
            server_aggregator=aggregator
        )
        logger.info("FedML server is running and waiting for clients...")
        server_runner.run()

    except Exception as e:
        logger.error(f"Error in run_server: {e}", exc_info=True)

def run_client(cli_args_obj):
    logger.info(f"Starting FedML client rank {cli_args_obj.client_id}")
    
    yaml_config = load_yaml_config_file(cli_args_obj.config)
    if not yaml_config:
        logger.error(f"[Client] Failed to load YAML config '{cli_args_obj.config}'.")
        return

    try:
        args = argparse.Namespace()
        
        args.data = argparse.Namespace(**yaml_config.get('data_args', {}))
        args.train = argparse.Namespace(**yaml_config.get('training_args', {}))
        args.device_args = argparse.Namespace(**yaml_config.get('device_args', {}))
        args.common = argparse.Namespace(**yaml_config.get('common_args', {}))
        args.env = argparse.Namespace(**yaml_config.get('environment_args', {}))
        
        args.rank = int(cli_args_obj.client_id)
        args.role = 'client'
        args.config_file = cli_args_obj.config
        args.comm = None
        args.federated_optimizer = args.train.federated_optimizer
        args.training_type = args.train.training_type
        args.scenario = args.env.scenario
        args.backend = args.env.backend
        args.worker_num = args.train.client_num_in_total if hasattr(args.train, 'client_num_in_total') else 2
        args.mqtt_host = args.env.mqtt_host
        args.mqtt_port = args.env.mqtt_port
        args.run_id = args.common.run_id
        
        client_trainer = FedTFLiteClient(client_id=args.rank, fedml_args=args)
        
        runner_class = getattr(fedml, 'FedMLRunner', None)
        if not runner_class:
            logger.error("[Client] FedMLRunner class not found.")
            return
        
        client_runner = runner_class(
            args=args,
            device=getattr(args, 'device', '/CPU:0'),
            dataset=[0, 0, None, None, {}, {}, {}, 0],
            model=None,
            client_trainer=client_trainer
        )
        
        logger.info(f"FedMLRunner for client rank {args.rank} starting...")
        client_runner.run()
        
        logger.info(f"Client {args.rank} is now running and waiting for server messages.")
        while True:
            time.sleep(1)
        
    except Exception as e:
        logger.error(f"Error in run_client: {e}", exc_info=True)
def main_cli_entry():
    parser = argparse.ArgumentParser(description='FedML IoV IDS')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    server_parser = subparsers.add_parser('server')
    server_parser.add_argument('--config', type=str, default='config/fedml_config.yaml')
    server_parser.add_argument('--rank', type=str, default='0', help='Server rank (should be 0)')
    
    client_parser = subparsers.add_parser('client')
    client_parser.add_argument('--client-id', type=str, required=True)
    client_parser.add_argument('--config', type=str, default='config/fedml_config.yaml')
    
    config_gen_parser = subparsers.add_parser('generate-config')
    config_gen_parser.add_argument('--output', type=str, default='config/fedml_config.yaml')

    parsed_cli_args = parser.parse_args()

    if parsed_cli_args.command == 'server':
        run_server(parsed_cli_args)
    elif parsed_cli_args.command == 'client':
        run_client(parsed_cli_args)
    elif parsed_cli_args.command == 'generate-config':
        generate_yaml_config_script(parsed_cli_args.output)
        
if __name__ == "__main__":
    main_cli_entry()