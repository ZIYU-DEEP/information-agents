import os
import yaml


def start_vllm(model: str='meta-llama/Llama-2-7b-chat-hf',
               tensor_parallel_size: int=4,
               port: int=8964):
    """
    Start the vllm server in the backend.
    """

    os.system(f"python -u -m vllm.entrypoints.openai.api_server \
                    --model {model} \
                    --tensor-parallel-size {tensor_parallel_size} \
                    --port {port} &")


if __name__ == '__main__':

    # Read the config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Set the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='gpt-3.5-turbo',
                        choices=['llama-2-7b', 'llama-2-13b', 'llama-2-70b'])
    parser.add_argument('--tensor_parallel_size', '-t', type=int, default=4)
    parser.add_argument('--port', '-p', type=int, default=8964)
    p = parser.parse_args()

    # Run the file
    start_vllm(config['MODEL'][p.model_name]['model'],
               p.tensor_parallel_size,
               p.port)
