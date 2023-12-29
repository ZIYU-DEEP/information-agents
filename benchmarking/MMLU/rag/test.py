"""
Using RAG to improve QA.
"""

# from openai import OpenAI
import ast  # for converting embeddings saved as strings back to arrays
import openai
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor
import asyncio
import argparse
import json

# ##########################################
# Helper Functions
# ##########################################
# ==========================================
# Get the prompt
# ==========================================
def get_prompt(task_data,
               task_name: str='anatomy',
               question_no: int=-1):
    '''
    Generate the 1-shot question prompt.

    Question num specifies which question will be used as prompt.
    If prompt_q is provided, it is used as 1-shot prompt question. This
    corresponds to GPT-4 based question prompts that we created.

    Else, we select question corresponding to question_num from the MMLU itself
    to generate the prompt. We select prompt from test set in this case,
    since train set is very small sometime and may not have 10 samples.
    We use 10 different prompts and take avergae over them to estimate
    performance on a subject.

    The function returns the 1-shot question prompt.
    '''

    prompt_set = 'test'

    prompt_add = f'This is a question from {task_name.replace("_", " ")}.\n'
    prompt_add += f'{task_data[prompt_set]["input"][question_no]}\n'

    for letter in ['A', 'B', 'C', 'D']:
        prompt_add += '    ' + letter + '. ' + task_data[prompt_set][letter][question_no] + '\n'

    prompt_add += f"Answer: "
    return prompt_add


# ==========================================
# Search function
# ==========================================
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100

) -> tuple[list[str], list[float]]:

    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    """

    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )

    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    return strings[:top_n], relatednesses[:top_n]


def num_tokens(text: str, model: str) -> int:
    """
    Return the number of tokens in a string.
    """

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# ==========================================
# RAG
# ==========================================
def query_message(query: str,
                  df: pd.DataFrame,
                  model: str,
                  token_budget: int) -> str:

    """
    Return a message for GPT, with relevant source texts pulled from a dataframe.
    """

    # Get strings ranked by relatedness
    strings, relatednesses = strings_ranked_by_relatedness(query, df, top_n=100)

    # Format the prompt
    message = (f'Use the below articles as an additional knowledge base'
               f'to answer the subsequent question.')
    question = f"\n\nQuestion: {query}"

    # Include the related strings in the prompt with naive filtration
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else: message += next_article

    return message + question


def rag_ask(query: str,
            model: str = 'gpt-3.5-turbo',
            df: pd.DataFrame = None,
            token_budget: int = 4096 - 500,
            print_message: bool = False,) -> str:
    """
    Answers a query using GPT and a dataframe of relevant texts and embeddings.
    """

    message = query_message(query, df, model=model, token_budget=token_budget)

    if print_message:
        print(message)
    messages = [
        {"role": "system",
         'content': f'You answer multiple choice questions.'
                    f'Please include only the letter choice in your answer and nothing else.'
                    f'Specifically, your response should only be one of A or B or C or D.'},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message


def plain_ask(query,
              model: str = 'gpt-3.5-turbo'):
    """
    Directly ask the query.
    """

    response = openai.ChatCompletion.create(
        messages=[{'role': 'system',
                   'content': f'You answer multiple choice questions.'},
                  {'role': 'user', 'content': query}],
        model=model,
        temperature=0)

    return response.choices[0].message.content


# ##########################################
# Main Function
# ##########################################
# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-tn', '--task_name', type=str, default='astronomy')
parser.add_argument('-ep', '--embeddings_root', type=str, default='../collections/wiki')
parser.add_argument('-em', '--EMBEDDING_MODEL', type=str, default='text-embedding-ada-002')
parser.add_argument('-gm', '--model', type=str, default='gpt-3.5-turbo')

# Parse arguments
p = parser.parse_args()

# Update globals
for key, value in vars(p).items():
    globals()[key] = value

result_path = f'{model}_{task_name}_rag_acc.json'
embeddings_path = f'{embeddings_root}/{task_name}.csv'

# Datasets
print('Loading dataset...')
task_data = load_dataset('lukaemon/mmlu', task_name)
print('Loading embeddings...')
df = pd.read_csv(embeddings_path)
df['embedding'] = df['embedding'].apply(ast.literal_eval)

# Function to process each task
def process_task(i, task_data, df, model):
    query = get_prompt(task_data, question_no=i)
    target = task_data['test']['target'][i]
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
