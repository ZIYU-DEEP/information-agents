"""
Using RAG to improve QA.
"""
from eval_utils import *
from openai import OpenAI
import ast  # for converting embeddings saved as strings back to arrays
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import asyncio
import argparse
import json


# ##########################################
# Arguments
# ##########################################
# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-t', '--task_name', type=str, default='prehistory')
parser.add_argument('-st', '--search_term', type=str, default='')
parser.add_argument('-ep', '--embeddings_path', type=str, default='')
parser.add_argument('-rr', '--result_root', type=str, default='./results')
parser.add_argument('-em', '--embedding_model', type=str, default='text-embedding-ada-002')  # Omitted
parser.add_argument('-gm', '--model', type=str, default='gpt-3.5-turbo')

# Parse arguments
p = parser.parse_args()

# Update globals
for key, value in vars(p).items():
    globals()[key] = value

# Automatically config the search term
if not search_term:
    if task_name.startswith('college_'):
        search_term = task_name.split('college_')[-1]
    else:
        search_term = task_name

# Automatically config the embeddings path
if not embeddings_path:
    embeddings_path = f'../collections/wiki/{search_term}.csv'



# ##########################################
# Main Function
# ##########################################
result_path = Path(result_root) / f'{model}_{task_name}_rag_acc.json'

# Datasets
print('Loading dataset...')
task_data = load_dataset('lukaemon/mmlu', task_name)

print('Loading embeddings...')
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Function to process each task
def process_task(i, task_data, df, model):
    # Get the query
    query = get_prompt(task_data, question_no=i)

    # Get the target value (A/B/C/D)
    target = task_data['test']['target'][i]

    # Calculate the accuracy
    acc_plain = int(target == plain_ask(query, model)[0])
    acc_rag = int(target == rag_ask(query, model, df)[0])
    return acc_plain, acc_rag

# Running the tasks in parallel using ThreadPoolExecutor
def run_parallel_tasks(task_data, df, model):

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_task, i, task_data, df, model)
                   for i in range(len(task_data['test']))]

        results = [future.result() for future in futures]

    return results

# Get the results
print('Run parallel tasks!')
results = run_parallel_tasks(task_data, df, model)

# Calculate the total accuracy
total_acc_plain = sum([result[0] for result in results])
total_acc_rag = sum([result[1] for result in results])

# Get the results
plain_acc = total_acc_plain / len(task_data['test'])
rag_acc = total_acc_rag / len(task_data['test'])

# Save the results to a JSON file
results_dict = {
    'task_name': task_name,
    'plain_acc': plain_acc,
    'rag_acc': rag_acc
}

with open(result_path, 'w') as json_file:
    json.dump(results_dict, json_file, indent=4)

print(results_dict)
print(f'Results saved to {result_path}.')
