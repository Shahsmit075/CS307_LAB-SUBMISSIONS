import numpy as np
import random
import json

# Utility to save data to a JSON file
def save_data_to_json(data, filename='data.json'):
    with open(filename, 'w') as outfile:
        json.dump(data.__dict__, outfile)

# Class representing a Menace player for tic-tac-toe
class MenaceTicTacToe:
    def __init__(self):
        self.matchboxes = {}
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.moves_played = []

    def save_progress(self):
        save_data_to_json(self)

# Check if a move is valid (between 0 and 8, and the spot is empty)
def is_valid_move(board, move):
    return 0 <= move <= 8 and board[move] == " "

# Get all available empty spaces on the board
def get_empty_spaces(board):
    return np.array([i for i in range(len(board)) if board[i] == ' '])

# Print the tic-tac-toe board
def display_board(board):
    print(f"\n 0 | 1 | 2     {board[0]} | {board[1]} | {board[2]}\n"
          f"---+---+---   ---+---+---\n"
          f" 3 | 4 | 5     {board[3]} | {board[4]} | {board[5]}\n"
          f"---+---+---   ---+---+---\n"
          f" 6 | 7 | 8     {board[6]} | {board[7]} | {board[8]}")

# Determine game outcome: 
# +10 for 'X' win
# -10 for 'O' win
# 0 for draw
# -1 for ongoing game

def check_game_status(board):
    # Check horizontal lines
    for i in range(0, 7, 3):  
        if board[i] == board[i+1] == board[i+2] != ' ':
            return 10 if board[i] == 'X' else -10
    # Check vertical lines
    for i in range(3):  
        if board[i] == board[i+3] == board[i+6] != ' ':
            return 10 if board[i] == 'X' else -10
    # Check diagonals
    if board[0] == board[4] == board[8] != ' ':
        return 10 if board[0] == 'X' else -10
    if board[2] == board[4] == board[6] != ' ':
        return 10 if board[2] == 'X' else -10
    # Check for draw
    if len(get_empty_spaces(board)) == 0:
        return 0
    return -1  # Game ongoing

# Get move for MENACE or human player
def get_next_move(board, player=None):
    # MENACE move
    if player:
        board_state = ''.join(board)
        if board_state not in player.matchboxes:
            available_moves = [i for i, v in enumerate(board) if v == ' ']
            player.matchboxes[board_state] = available_moves * ((len(available_moves) + 2) // 2)
        beads = player.matchboxes[board_state]
        move = random.choice(beads) if beads else -1
        player.moves_played.append((board_state, move))
        return move
    # Human move
    else:  
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if is_valid_move(board, move):
                    return move
                print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid number.")

# Update MENACE's data based on game result
def update_menace(player, result):
    if result == "win":
        for board, move in player.moves_played:
            player.matchboxes[board].extend([move] * 3)
        player.wins += 1
    elif result == "lose":
        for board, move in player.moves_played:
            player.matchboxes[board].remove(move)
        player.losses += 1
    elif result == "draw":
        for board, move in player.moves_played:
            player.matchboxes[board].append(move)
        player.draws += 1
    player.save_progress()

# Train MENACE against itself
def train_menace(player1, player2, num_games=1000):
    for _ in range(num_games):
        board = [' '] * 9
        player1.moves_played = []
        player2.moves_played = []
        while check_game_status(board) == -1:
            move = get_next_move(board, player1)
            board[move] = 'O'
            if check_game_status(board) != -1:
                break
            move = get_next_move(board, player2)
            board[move] = 'X'
        outcome = check_game_status(board)
        if outcome == 10:
            update_menace(player1, "lose")
        elif outcome == -10:
            update_menace(player1, "win")
        else:
            update_menace(player1, "draw")

# Initialize MENACE player
menace_player = MenaceTicTacToe()

# Load saved data if it exists, else train MENACE
try:
    with open('data.json', 'r') as infile:
        saved_data = json.load(infile)
        menace_player.matchboxes = saved_data.get("matchboxes", {})
        menace_player.wins = saved_data.get("wins", 0)
        menace_player.losses = saved_data.get("losses", 0)
        menace_player.draws = saved_data.get("draws", 0)
except FileNotFoundError:
    second_player = MenaceTicTacToe()
    train_menace(menace_player, second_player)
    print("No previous game data found. Training completed.")

# Play against MENACE
def play_game():
    board = [' '] * 9
    display_board(board)
    first_player = input("Would you like to go first? (Y/N): ").lower().startswith('y')
    if first_player:
        print("You are O")
        while check_game_status(board) == -1:
            move = get_next_move(board)  # Human move
            board[move] = 'O'
            display_board(board)
            if check_game_status(board) != -1:
                break
            move = get_next_move(board, menace_player)  # MENACE move
            board[move] = 'X'
            display_board(board)
    else:
        print("You are X")
        while check_game_status(board) == -1:
            move = get_next_move(board, menace_player)  # MENACE move
            board[move] = 'O'
            display_board(board)
            if check_game_status(board) != -1:
                break
            move = get_next_move(board)  # Human move
            board[move] = 'X'
            display_board(board)

    outcome = check_game_status(board)
    if outcome == 10:
        update_menace(menace_player, "lose")
        print("You lost!")
    elif outcome == -10:
        update_menace(menace_player, "win")
        print("You won!")
    else:
        update_menace(menace_player, "draw")
        print("It's a draw!")

play_game()
