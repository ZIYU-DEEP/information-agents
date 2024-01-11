"""
Using RAG to improve QA.
"""

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
# Helper Functions
# ##########################################
# ==========================================
# Get the prompt
# ==========================================
def get_prompt(task_data,
               task_name: str='anatomy',
               question_no: int=-1):
    '''
    Format the prompt from the MMLU dataset.
    '''

    prompt_set = 'test'

    prompt_add = f'This is a question from {task_name.replace("_", " ")}.\n'
    prompt_add += f'{task_data[prompt_set]["input"][question_no]}\n'

    for letter in ['A', 'B', 'C', 'D']:
        prompt_add += '    ' + letter + '. ' + task_data[prompt_set][letter][question_no] + '\n'

    prompt_add += (f'Now please provide your answer.'
                   f'Include only the letter choice in your answer and nothing else.'
                   f'Specifically, your response should only be one of A or B or C or D.')

    prompt_add += f'Your answer: '
    return prompt_add


# ==========================================
# Search function
# ==========================================
def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100,
    embedding_model: str = 'text-embedding-ada-002'

) -> tuple[list[str], list[float]]:

    """
    Returns a list of strings and relatednesses, sorted from most related to least.
    """

    query_embedding_response = OpenAI().embeddings.create(
        model=embedding_model,
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
                  token_budget: int,
                  embedding_model: str) -> str:

    """
    Return a message for GPT, with relevant source texts pulled from a dataframe.
    """

    # Get strings ranked by relatedness
    strings, relatednesses = strings_ranked_by_relatedness(
        query=query,
        df=df,
        top_n=100,
        embedding_model=embedding_model)

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
            print_message: bool = False,
            embedding_model: str='text-embedding-ada-002') -> str:
    """
    Answers a query using GPT and a dataframe of relevant texts and embeddings.
    """

    message = query_message(
        query=query,
        df=df,
        model=model,
        token_budget=token_budget,
        embedding_model=embedding_model)

    if print_message:
        print(message)
    messages = [
        {'role': 'system',
         'content': f'You answer multiple choice questions.'
                    f'Please include only the letter choice in your answer and nothing else.'
                    f'Specifically, your response should only be one of A or B or C or D.'},
        {'role': 'user',
        'content': message},
    ]
    response = OpenAI().chat.completions.create(
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

    response = OpenAI().chat.completions.create(
        messages=[
        {'role': 'system',
         'content': f'You answer multiple choice questions.'
                    f'Please include only the letter choice in your answer and nothing else.'
                    f'Specifically, your response should only be one of A or B or C or D.'},
        {'role': 'user',
         'content': query},
        ],
        model=model,
        temperature=0)

    return response.choices[0].message.content
