from unicodedata import bidirectional
from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

def Xor(*sentences):
    return And(Or(*sentences), Not(And(*sentences)))

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # Given by definition
    Xor(AKnight, AKnave),
     
    # Given by the character
    ## A says "I am both a knight and a knave."
    Biconditional(AKnight, And(AKnight, AKnave)),
    Biconditional(AKnave, Not(And(AKnight, AKnave)))
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # Given by definition
    Xor(AKnight, AKnave),
    Xor(BKnight, BKnave),

    # Given by the character
    ## A says "We are both knaves."
    Biconditional( AKnight, And(AKnave, BKnave) ),
    Biconditional( AKnave, Not(And(AKnave, BKnave)) )
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # Given by definition
    Xor(AKnight, AKnave),
    Xor(BKnight, BKnave),

    # Given by the characters
    ## A says "We are the same kind."
    Biconditional( AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave)) ),
    Biconditional( AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave))) ),
    ## B says "We are of different kinds."
    Biconditional( BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight)) ),
    Biconditional( BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight))) )
)

AFirst = Symbol("A said 'I am a knight'")
ASecond = Symbol("A said 'I am a knave'")

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # Given by definition
    Xor(AKnight, AKnave),
    Xor(BKnight, BKnave),
    Xor(CKnight, CKnave),

    # Given by the characters
    ## A says either "I am a knight." or "I am a knave.", but you don't know which.
    Xor(AFirst, ASecond),
    Biconditional( AFirst, And(Implication(AKnight, AKnight), Implication(AKnave, Not(AKnight))) ),
    Biconditional( ASecond, And(Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))) ),
    ## B says "A said 'I am a knave'."
    Biconditional( BKnight, ASecond ),
    Biconditional( BKnave, Not(ASecond) ),
    ## B says "C is a knave."
    Biconditional( BKnight, CKnave ),
    Biconditional( BKnave, Not(CKnave) ),
    ## C says "A is a knight."
    Biconditional( CKnight, AKnight ),
    Biconditional( CKnave, Not(AKnight) )
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
