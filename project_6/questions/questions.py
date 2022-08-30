import nltk
import sys
import os
import string
import math


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))


    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        _, file_ext = os.path.splitext(file_name)

        if file_ext == ".txt":
            with open(file_path, "r", encoding="utf8") as file:
                corpus[file_name] = file.read()

    return corpus



def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = [token.lower() for token in nltk.tokenize.word_tokenize(document)]
    tokens = [token for token in tokens if (token not in string.punctuation) and (token not in nltk.corpus.stopwords.words("english"))]

    return tokens


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Returns the number of documents that contain a given word
    def occurrenciesOf(word):
        times = 0
        for filename in documents:
            if word in documents[filename]: times += 1
        return times

    words_idf = {}

    for filename in documents:
        for word in documents[filename]:
            if not word in words_idf: # Compute idf only for unknown words
                words_idf[word] = math.log( len(documents) / occurrenciesOf(word) )

    return words_idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    ranking = [] # Files saved as (filename, tf-idf sum)

    # Computing tf-idf sum for each file
    for filename in files:
        rank = 0
        for word in query:
            if word in files[filename]:
                rank += files[filename].count(word) * idfs[word]

        ranking.append( (filename, rank) )
    
    ranking.sort(key=lambda rank: rank[1], reverse=True)
    return [filename for filename, _ in ranking[:n]]
    


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    ranking = [] # Files saved as (filename, idf sum, query term density)

    for sentence in sentences:
        query_words = [word for word in query if word in sentences[sentence]]
        
        # Idfs sum
        rank = 0
        for word in query_words: rank += idfs[word]
        
        # Query term density
        sentence_query_words_count = len([word for word in sentences[sentence] if word in query])
        query_term_density = sentence_query_words_count / len(sentences[sentence])

        ranking.append( (sentence, rank, query_term_density) )

    ranking.sort(key=lambda rank: (rank[1], rank[2]), reverse=True)
    return [sentence for sentence, _, _ in ranking[:n]]

if __name__ == "__main__":
    main()
