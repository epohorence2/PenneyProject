import numpy as np 
import os 
from datetime import datetime
from src.utils import time_and_size


class DeckStack: 
    def __init__(self, num_decks=2_000_000, batch_size=10_000, out_dir="decks", seed=12):
        self.num_decks = num_decks
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.rng = np.random.default_rng(seed) 
        os.makedirs(out_dir, exist_ok=True)
    
    @time_and_size
    def generate_and_store_decks(self):
        """
        Generate and store shuffled decks of 52 cards with 26 red (0) and 26 black (1) cards
        """
        deck = np.array([0] * 26 + [1] * 26)
        num_batches = self.num_decks // self.batch_size

        for batch_idx in range(num_batches):
            # Generate a batch of shuffled decks
            decks = np.empty((self.batch_size, 52), dtype=np.uint8)
            for i in range(self.batch_size):
                decks[i] = self.rng.permutation(deck)

            # Save the batch to a file
            batch_file = os.path.join(self.out_dir, f"decks_batch_{batch_idx + 1}.npy")
            np.save(batch_file, decks)
            print(f"Saved batch {batch_idx + 1}/{num_batches} to {batch_file}")

        print("All decks generated and stored.")

    
deck_stack = DeckStack(num_decks=2_000_000, batch_size=10_000, out_dir="decks", seed=12)
deck_stack.generate_and_store_decks()

    # def time_and_size(self, func): 
    #     def decorator(func):
    #         def wrapper(*args, **kwargs):
    #             start_time = datetime.now()
    #             result = func(*args, **kwargs)
    #             end_time = datetime.now()
    #             duration = (end_time - start_time).total_seconds()
    #             size = os.path.getsize(result) if isinstance(result, str) and os.path.exists(result) else None
    #             print(f"Function '{func.__name__}' executed in {duration:.2f} seconds.")
    #             if size is not None:
    #                 print(f"Output file size: {size / (1024 * 1024):.2f} MB.")
    #             return result
    #         return wrapper
    #     return decorator 

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

