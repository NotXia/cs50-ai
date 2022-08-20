from distutils.log import warn
import sys
from queue import Queue

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
            for word in list(self.domains[variable]):
                if len(word) != variable.length: 
                    self.domains[variable].remove(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        def searchWordWithCharacterAtPosition(words, character, position):
            for word in words:
                if word[position] == character: return True
            return False

        overlap = self.crossword.overlaps[x, y]
        revised = False

        if overlap != None:
            for word in list(self.domains[x]):
                # Remove from X's domain all the words that are non compatible with Y
                if not searchWordWithCharacterAtPosition(self.domains[y], word[overlap[0]], overlap[1]): 
                    self.domains[x].remove(word)
                    revised = True
                
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        queue = Queue()

        if arcs != None:
            for arc in arcs: queue.put(arc)
        else:
            for variable in self.crossword.variables:
                for neighbor in self.crossword.neighbors(variable):
                    queue.put( (variable, neighbor) )
        
        while not queue.empty():
            x, y = queue.get()
            if self.revise(x, y):
                if len(self.domains[x]) == 0: return False
                for neighbor in self.crossword.neighbors(x):
                    if neighbor == y: continue
                    if not (neighbor, x) in queue.queue: queue.put( (neighbor, x) )
        
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        return len(self.crossword.variables) == len(assignment)

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = set()

        for variable in assignment:
            # Word length check
            if variable.length != len(assignment[variable]): return False 

            # Overlapping character check
            for neighbor in self.crossword.neighbors(variable):
                if neighbor in assignment:
                    index1, index2 = self.crossword.overlaps[variable, neighbor]
                    if assignment[variable][index1] != assignment[neighbor][index2]: return False

            words.add(assignment[variable])
        
        # Words uniqueness check
        if len(words) != len(assignment): return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        domain = []
        interesting_neighbors = [neighbor for neighbor in self.crossword.neighbors(var) if not (neighbor in assignment)] # Ignore neighbors with an assigned word

        for word in self.domains[var]:
            eliminated_choices = 0
            # Count ruled out neighbors
            for neighbor in interesting_neighbors:
                if word in self.domains[neighbor]: eliminated_choices += 1
            domain.append( (word, eliminated_choices) )
        domain.sort(key=lambda val: val[1])
        
        return [word for word, _ in domain]

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        interesting_variables = [variable for variable in self.crossword.variables if not (variable in assignment)]

        curr_best = interesting_variables[0]
        interesting_variables = interesting_variables[1::]

        for variable in interesting_variables:
            if ((len(self.domains[variable]) < len(self.domains[curr_best])) or 
            ((len(self.domains[variable]) == len(self.domains[curr_best])) and (len(self.crossword.neighbors(variable)) > len(self.crossword.neighbors(curr_best))))): 
                curr_best = variable

        return curr_best

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment): return assignment

        to_solve_variable = self.select_unassigned_variable(assignment)

        for word in self.order_domain_values(to_solve_variable, assignment):
            assignment[to_solve_variable] = word

            if self.consistent(assignment):
                self.ac3([ (neighbor, to_solve_variable) for neighbor in self.crossword.neighbors(to_solve_variable) ]) # Inference
                result = self.backtrack(assignment)
                if result != None: return result
                
            del assignment[to_solve_variable]

        return None

def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
