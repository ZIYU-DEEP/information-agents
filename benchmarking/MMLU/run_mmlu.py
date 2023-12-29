from litellm import acompletion
from tqdm import tqdm
from utils import *
from typing import Dict
import pandas as pd
import asyncio
import litellm
import json
import yaml
import argparse
import os


async def get_response(prompt: str, model: str, config: Dict):
    """
    Get the response from the model.
    """

    # Extract the model config
    model_config = config['MODEL'][model]

    # Construct messages with the dynamic prompt
    messages = [
        {'role': 'system',
         'content': f'Follow the given examples and answer the question.'
                    f'Please include only the letter choice in your answer and nothing else.'
                    f'Specifically, your response should only be one of A or B or C or D.'},  # This is temporary

        {'role': 'user',
        'content': prompt}
    ]

    # Add the messages to the model_config
    model_config['messages'] = messages

    # Call acompletion with the unpacked config
    response = await acompletion(**model_config)

    return response


def main(args, tasks=TASKS):
    if 'gpt' in args.model_name:
        if args.openai_api_key:
            os.environ['OPENAI_API_KEY'] = args.openai_api_key

    elif 'gemini' in args.model_name:
        # gemini evaluation
        litellm.vertex_project = ""  # Your Project ID
        litellm.vertex_location = ""  # Your Project Location
        litellm.drop_params = True

    if args.cot:
        mmlu_cot_prompt = json.load(open('data/mmlu-cot.json'))
        method_name = 'cot'
    else:
        method_name = 'simple'

    # Initialize for the accuracy
    all_acc, all_number, accs_json = 0, 0, {}

    # Set the file path
    os.makedirs('outputs', exist_ok=True)
    outputs_file = open(f'outputs/{args.model_name}_{method_name}_outputs.json', 'a')

    # Load the config
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    for task in tasks:
        print(f'Testing {task} ...')

        # Initialize the accuracy
        acc = 0

        # Reading files
        dev_df = pd.read_csv(
            os.path.join('data', 'dev', task + '_dev.csv'), header=None
        )[: args.num_examples]

        test_df = pd.read_csv(
            os.path.join('data', 'test', task + '_test.csv'), header=None
        )

        for i in tqdm(range(test_df.shape[0])):

            # CoT prompting
            if args.cot:

                # Format the question prompt
                q = test_df.iloc[i, 0] + '\n'
                for j, letter in enumerate(['A', 'B', 'C', 'D']):
                    q += '(' + letter + ') ' + str(test_df.iloc[i, j + 1]) + ' '

                # Add THE CoT prompt, each are about 5 shot examples
                q += "\nA: Let's think step by step."
                prompt = mmlu_cot_prompt[task] + '\n\n' + q
                label = test_df.iloc[i, test_df.shape[1] - 1]

                # Get the response
                response = asyncio.run(get_response(prompt, args.model_name, config))
                ans_model = response["choices"][0]["message"]["content"]
                ans_, residual = extract_ans(ans_model)

                # Calculate the accuracy
                ans_model, correct = test_answer_mmlu_(ans_, label)
                if correct: acc += 1

            # Simple prompting
            else:
                # Set the number for the examples
                k = args.num_examples

                # Format the prompt
                prompt_end = format_example(test_df, i, include_answer=False)
                train_prompt = gen_prompt(dev_df, task, k)
                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1] - 1]

                # Get the response
                response = asyncio.run(get_response(prompt, args.model_name, config))
                # 0 means the answer character [A, B, C, D] (sometimes model will output more)
                ans_model = response["choices"][0]["message"]["content"][0]

                # Calculate the accuracy
                correct = ans_model == label
                if correct: acc += 1

            # Write the results
            outputs_file.write(
                json.dumps(
                    {
                        "task": task,
                        "correct": correct,
                        "prediction": ans_model,
                        "label": label,
                        "response": response["choices"][0]["message"]["content"],
                        "question": test_df.iloc[i, 0],
                        "A": test_df.iloc[i, 1],
                        "B": test_df.iloc[i, 2],
                        "C": test_df.iloc[i, 3],
                        "D": test_df.iloc[i, 4],
                        "prompt": prompt,
                    },
                    indent=4,
                )
                + "\n"
            )
        print("%s acc %.4f" % (task, acc / test_df.shape[0]))
        accs_json[task] = acc / test_df.shape[0]
        all_acc += acc
        all_number += test_df.shape[0]

    # Save the results
    accs_json["all"] = all_acc / all_number
    json.dump(accs_json, open(f"outputs/{args.model_name}_{method_name}_accs.json", "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--model_name', type=str,
                        default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-4-1106-preview',
                                 'gemini-pro', 'mixtral',
                                 'llama-2-7b', 'llama-2-13b', 'llama-2-70b'])
    parser.add_argument('--cot', action='store_true')
    parser.add_argument('--num_examples',
                        type=int,
                        default=5,
                        help='Number of examples included in the prompt input.')
    args = parser.parse_args()
    main(args)
