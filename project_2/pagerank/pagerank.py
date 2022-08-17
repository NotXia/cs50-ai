import os
import random
import re
import sys
import numpy as np
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability = { page_name: 0 for page_name in corpus }

    if len(corpus[page]) > 0:
        # Exploring neighbours
        for neighbour in corpus[page]:
            probability[neighbour] += (damping_factor / len(corpus[page]))

        # Exploring a random page
        for page_name in corpus:
            probability[page_name] += ((1-damping_factor) / len(corpus))
    else:
        for page_name in corpus:
            probability[page_name] += (1 / len(corpus))


    return probability


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    visited_times = { page_name: 0 for page_name in corpus }

    # First sample
    curr_page = random.choice( list(corpus.keys()) )
    visited_times[curr_page] += 1

    for _ in range(n-1):
        next_probability = transition_model(corpus, curr_page, damping_factor)

        # Next page
        curr_page = np.random.choice(list(next_probability.keys()), p=list(next_probability.values()))
        visited_times[curr_page] += 1

    return { page_name: (visited_times[page_name] / n) for page_name in corpus }


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def convergence(old_probability, new_probability):
        diff = [ math.fabs(old_probability[page_name]-new_probability[page_name]) for page_name in corpus ]
        return all( x <= 0.001 for x in diff )

    probability = { page_name: 1/len(corpus) for page_name in corpus }
    prev_probability = { page_name: -float("inf") for page_name in corpus }
    
    while not convergence(prev_probability, probability):
        prev_probability = probability.copy()

        for curr_page in corpus:
            # Determining the pages (and their neighbours) that links to the current page
            links_to_me = {}
            for page_name in corpus:
                if curr_page in corpus[page_name] or len(corpus[page_name]) == 0: 
                    links_to_me[page_name] = corpus[page_name] if len(corpus[page_name]) > 0 else list(corpus.keys())

            probability[curr_page] = ((1-damping_factor)/len(corpus)) + (damping_factor*sum(prev_probability[neighbour]/len(links_to_me[neighbour]) for neighbour in links_to_me))
    
    return probability


if __name__ == "__main__":
    main()
