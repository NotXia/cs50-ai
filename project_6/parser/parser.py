import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> Phr
Phr -> NP VP | NP VP NP | VP NP
Phr -> Phr Conj Phr | Phr Adv
NP -> N | Det N | AL N | Det AL N
NP -> P NP | NP P NP
AL -> Adj | Adj AL
VP -> V | Adv V | V Adv
"""
# NP: Noun phrase
# AL: Adjective List
# VP: Verb phrase


grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    def containsLetter(word):
        for letter in word:
            if letter.isalpha(): return True
        return False

    tokens = nltk.tokenize.word_tokenize(sentence)
    tokens = [ token.lower() for token in tokens if containsLetter(token) ]

    return tokens


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    chunks = []

    # Returns True if a given tree contains a child with label symbol. False otherwise.
    def hasSymbolAsChild(tree, symbol):
        for child in tree:
            if type(child) == nltk.tree.Tree:
                if child.label() == symbol: 
                    return True
                else:
                    return hasSymbolAsChild(child, symbol)
        return False

    def walk(tree):
        for child in tree:
            if type(child) == nltk.tree.Tree:
                if child.label() == "NP" and not hasSymbolAsChild(child, "NP"): 
                    chunks.append(child)
                else:
                    walk(child)

    walk(tree)

    return chunks


if __name__ == "__main__":
    main()
