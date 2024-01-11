"""
Crawling from Wikipedia.
c.r. OpenAI Cookbook
"""
from tqdm import tqdm
from openai import OpenAI  # for generating embeddings

import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import argparse


# ##########################################
# Helper Functions
# ##########################################
# ==========================================
# Search the relevant category
# ==========================================
def search_categories(site, search_term):
    """
    Search for categories using the search API.
    """
    categories = []

    for page in tqdm(site.search(search_term, namespace=14),
                     desc=f"Searching for '{search_term}'"):

        categories.append(page['title'])
    return categories


def all_categories(site: mwclient.Site) -> list[str]:
    """
    Return a list of all category names on a given Wiki site.
    """

    categories = []
    for category in site.allcategories():
        categories.append(category.name)
        if len(categories) > 100:
            break
    return categories


# ==========================================
# Search the relevant category
# ==========================================
def titles_from_category(category: mwclient.listing.Category,
                         max_depth: int,
                         limit: int=3000) -> set[str]:
    """
    Return a set of page titles in a given Wiki category and its subcategories.
    """

    titles = set()

    for cm in tqdm(category.members()):
        if type(cm) == mwclient.page.Page:
            # ^type() used instead of isinstance() to catch match w/ no inheritance
            titles.add(cm.name)
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)
            titles.update(deeper_titles)

        if limit:
            if len(titles) > limit:
                return titles

    return titles


# ==========================================
# Chunk the documents
# ==========================================
# define functions to split Wikipedia pages into sections
SECTIONS_TO_IGNORE = ["See also", "References", "External links",
                      "Further reading", "Footnotes", "Bibliography",
                      "Sources", "Citations", "Literature", "Footnotes",
                      "Notes and references", "Photo gallery",
                      "Works cited", "Photos", "Gallery", "Notes",
                      "References and sources", "References and notes"]


def all_subsections_from_section(section: mwparserfromhell.wikicode.Wikicode,
                                 parent_titles: list[str],
                                 sections_to_ignore: set[str],
                                 ) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia section, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """

    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        # ^wiki headings are wrapped like "== Heading =="
        return []

    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]

    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]

        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection,
                                                        titles,
                                                        sections_to_ignore))

        return results


def all_subsections_from_title(title: str,
                               sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,
                               site_name: str = 'en.wikipedia.org',
                               ) -> list[tuple[list[str], str]]:
    """
    From a Wikipedia page title, return a flattened list of all nested subsections.
    Each subsection is a tuple, where:
        - the first element is a list of parent subtitles, starting with the page title
        - the second element is the text of the subsection (but not any children)
    """

    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]

    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)

    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection,
                                                    [title],
                                                    sections_to_ignore))
    return results


# ==========================================
# Clearn the texts and filtering
# ==========================================
# clean text
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    """
    Return a cleaned up section with:
        - <ref>xyz</ref> patterns removed
        - leading/trailing whitespace removed
    """
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)
    text = text.strip()
    return (titles, text)


# filter out short/blank sections
def keep_section(section: tuple[list[str], str]) -> bool:
    """Return True if the section should be kept, False otherwise."""
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True

# ==========================================
# Tokenization
# ==========================================
def num_tokens(text: str,
               model: str='gpt-3.5-turbo') -> int:

    """
    Return the number of tokens in a string.
    """

    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    """
    Split a string in two, on a delimiter, trying to balance tokens on each side.
    """

    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found

    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point

    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)

            if diff >= best_diff:
                break
            else:
                best_diff = diff

        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(string: str,
                     model: str,
                     max_tokens: int,
                     print_warning: bool = True) -> str:
    """
    Truncate a string to a maximum number of tokens.
    """

    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])

    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")

    return truncated_string


def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = 'gpt-3.5-turbo',
    max_recursion: int = 5,
) -> list[str]:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """

    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)

    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]

    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]

    # otherwise, split in half and recurse
    else:
        titles, text = subsection

        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)

            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue

            else:
                # recurse on each half
                results = []

                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, model=model, max_tokens=max_tokens)]
