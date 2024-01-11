"""
Crawling from Wikipedia.
c.r. OpenAI Cookbook
"""
from wiki_utils import *
from tqdm import tqdm
from openai import OpenAI  # for generating embeddings

import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import argparse


# ##########################################
# Arguments
# ##########################################
# Create the parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-t', '--task_name', type=str, default='prehistory')
parser.add_argument('-st', '--search_term', type=str, default='')
parser.add_argument('-ep', '--embeddings_path', type=str, default='')
parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo')
parser.add_argument('-ws', '--wiki_site', type=str, default='en.wikipedia.org')
parser.add_argument('-mt', '--max_tokens', type=int, default=1600)
parser.add_argument('-em', '--embedding_model', type=str, default='text-embedding-ada-002')
parser.add_argument('-bs', '--batch_size', type=int, default=1000)
parser.add_argument('-cl', '--category_limit', type=int, default=0,
                    help='0 means no limit; otherwise the number means number of titles to download.')

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
# Main function
# ##########################################
# ==========================================
# 0. Search the relevant category
# ==========================================
site = mwclient.Site(wiki_site)
found_categories = search_categories(site, search_term)
CATEGORY_TITLE = found_categories[0]  # Notice here we simply get the very first category title

# ==========================================
# 1. Search the relevant articles from the cate
# ==========================================
site = mwclient.Site(wiki_site)
category_page = site.pages[CATEGORY_TITLE]
titles = titles_from_category(category_page, max_depth=1, limit=category_limit)
# ^note: max_depth=1 means we go one level deep in the category tree
print(f"Found {len(titles)} article titles in {CATEGORY_TITLE}.")

# ==========================================
# 2. Chunk the documents
# ==========================================
# split pages into sections
# may take ~1 minute per 100 articles
wikipedia_sections = []
for title in tqdm(titles):
    wikipedia_sections.extend(all_subsections_from_title(title))
print(f"Found {len(wikipedia_sections)} sections in {len(titles)} pages.")

# ==========================================
# 3. Clearn the texts and filtering
# ==========================================
# clean text
wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]
original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]
print(f"Filtered out {original_num_sections-len(wikipedia_sections)} sections")
print(f"leaving {len(wikipedia_sections)} sections.")

# ==========================================
# 4. Chunk and Save
# ==========================================
# split sections into chunks
wikipedia_strings = []
for section in tqdm(wikipedia_sections):
    wikipedia_strings.extend(split_strings_from_subsection(section, max_tokens=max_tokens))

print(f'{len(wikipedia_sections)} Wikipedia sections split into {len(wikipedia_strings)} strings.')

# Calculate Embeddings
embeddings = []
client = OpenAI()

for batch_start in tqdm(range(0, len(wikipedia_strings), batch_size)):
    batch_end = batch_start + batch_size
    batch = wikipedia_strings[batch_start:batch_end]
    print(f"Batch {batch_start} to {batch_end-1}")
    response = client.embeddings.create(model=embedding_model, input=batch)
    for i, be in enumerate(response.data):
        assert i == be.index  # double check embeddings are in same order as input
    batch_embeddings = [e.embedding for e in response.data]
    embeddings.extend(batch_embeddings)

df = pd.DataFrame({'text': wikipedia_strings, 'embedding': embeddings})
df.to_csv(embeddings_path, index=False)
