import os
import yaml
import argparse


def start_vllm(model: str = 'meta-llama/Llama-2-7b-chat-hf',
               tensor_parallel_size: int = 4,
               port: int = 8964,
               cuda_devices: str = '1,2,3,4'):
    """
    Start the vllm server in the backend.
    """

    cuda_str = f"CUDA_VISIBLE_DEVICES={cuda_devices}" if cuda_devices else ""
    os.system(f"{cuda_str} python -u -m vllm.entrypoints.openai.api_server \
                    --model {model} \
                    --tensor-parallel-size {tensor_parallel_size} \
                    --port {port} &")


if __name__ == '__main__':

    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='gpt-3.5-turbo',
                        choices=['llama-2-7b', 'llama-2-13b', 'llama-2-70b'])
    parser.add_argument('--port', '-p', type=int, default=8964)
    parser.add_argument('--cuda_devices', '-c', type=str, default=None,
                        help='Optional: Comma-separated list of CUDA device IDs')
    p = parser.parse_args()

    # Read the config
    with open('config.yaml', 'r') as file:
        server_config = yaml.safe_load(file)['SERVER'][p.model_name]

    # Run the file
    start_vllm(server_config['model'],
               server_config['tensor_parallel_size'],
               p.port,
               p.cuda_devices)
