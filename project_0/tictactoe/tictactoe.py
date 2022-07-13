"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = o_count = 0

    for row in board:
        x_count += row.count(X)
        o_count += row.count(O)
    
    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = []

    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell is EMPTY:
                moves.append((i, j))
    
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if (not 0 <= action[0] <= 2) or (not 0 <= action[1] <= 2) or (board[action[0]][action[1]] != EMPTY):
        raise Exception("Invalid move")

    new_board = deepcopy(board)
    new_board[action[0]][action[1]] = player(board)

    return new_board


def winnerOfLine(line):
    """
    Returns the winner, if there is one, of a line of the board.
    """
    if line.count(X) == 3 or line.count(O) == 3:
        return line[0]
    else:
        return None


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    
    for pos in range(0, 3):
        row = [board[pos][j] for j in range(0, 3)]
        column = [board[i][pos] for i in range(0, 3)]
        
        # Row checking
        winner = winnerOfLine(row)
        if winner != None:
            return winner

        # Column checking
        winner = winnerOfLine(column)
        if winner != None:
            return winner

    # Diagonal checking
    if (board[0][0] == board[1][1] == board[2][2] != EMPTY) or (board[0][2] == board[1][1] == board[2][0] != EMPTY):
        return board[1][1]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) != None:
        return True
    
    # Checks for a non-EMPTY cell
    for row in board:
        if row.count(EMPTY) > 0:
            return False
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)

    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    else:
        return 0


def propagate(board, alpha, beta):
    """
    Executes minimax returning the tuple (score, move)
    
    alpha keeps track of the best maximized score
    beta keeps track of the best minimized score
    """
    if terminal(board):
        return (utility(board), None)

    # Maximize
    if player(board) == X:
        best_score, best_move = -math.inf, None
        for move in actions(board):
            score, _ = propagate(result(board, move), alpha, beta)
            if max(best_score, score) != best_score:
                best_score, best_move = score, move
                alpha = best_score
                # Stop if the current best of the branch is better than the previously known one (this move is surely better)
                if best_score >= beta: break
    # Minimize
    else:
        best_score, best_move = math.inf, None
        for move in actions(board):
            score, _ = propagate(result(board, move), alpha, beta)
            if min(best_score, score) != best_score:
                best_score, best_move = score, move
                beta = best_score
                # Stop if the previously known score is better than the current best of this branch (the previous move is surely better)
                if alpha >= best_score: break

    return (best_score, best_move)


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    score, move = propagate(board, -math.inf, math.inf)
    return move
        
