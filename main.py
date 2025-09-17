import numpy as np 
import os 


    # def check_wins(self, deck, sequence):
    #     """
    #     Count the number of wins for each player 
    #     """
    #     count = 0
    #     for i in range(len(deck) - len(sequence) + 1):
    #         if np.array_equal(deck[i:i+len(sequence)], sequence):
    #             count += 1
    #     return count
    
    # def winner(self, deck, player1, player2):
    #     """
    #     Check total hits for each sequence through scanning the full deck
    #     """
    #     wins1 = self.check_wins(deck, player1)
    #     wins2 = self.check_wins(deck, player2)
    #     if wins1 > wins2:
    #         return 1 
    #     elif wins2 > wins1:
    #         return 2
    #     else:
    #         return 0

    # def play_all_decks(self, player1, player2):
    #     """
    #     Load generated decks and track winners across all batches
    #     """
    #     results = {1: 0, 2: 0, 0: 0}  # player1 wins, player2 wins, ties
    #     batch_files = [f for f in os.listdir(self.out_dir) if f.endswith(".npy")]

    #     for bf in batch_files:
    #         decks = np.load(os.path.join(self.out_dir, bf))
    #         for deck in decks:
    #             w = self.winner(deck, player1, player2)
    #             results[w] += 1

    #     print(f'Player 1 Wins: {results[1]}, Player 2 Wins: {results[2]}, Ties: {results[0]}')
    #     return results

