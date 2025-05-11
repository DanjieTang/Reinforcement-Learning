import torch
from collections import deque

class SnakeGame:
    def __init__(self, board_size: int = 16):
        self.board_size: int = board_size
        self.board: torch.Tensor = torch.zeros(board_size, board_size)
        self.snake_body: deque[tuple[int, int]] | None = None
        self.food_location: tuple[int, int] | None = None

    def init_game(self):
        self.board: torch.Tensor = torch.zeros(self.board_size, self.board_size)

        # Randomly place the snake within [1/4, 3/4] of the board
        snake_head_location: tuple[int, int] = (
            torch.randint(self.board_size // 4, self.board_size * 3 // 4, (1,)).item(),
            torch.randint(self.board_size // 4, self.board_size * 3 // 4, (1,)).item(),
        )
        self.board[snake_head_location] = -0.5

        # Randomly place the food within the board
        self.food_location: tuple[int, int] = (
            torch.randint(0, self.board_size, (1,)).item(),
            torch.randint(0, self.board_size, (1,)).item(),
        )
        self.board[self.food_location] = 1

        # Initialize the queue that represents the snake body
        self.snake_body: deque[tuple[int, int]] = deque([snake_head_location], maxlen=1)

    def take_action(self, action: int) -> float:
        """
        Take an action and return the reward.

        :param action: The action to take. 0 = up, 1 = down, 2 = left, 3 = right.
        :return: The reward.
        """
        next_location: tuple[int, int] | None = None
        if action == 0:
            # Up
            next_location = (self.snake_body[0][0] - 1, self.snake_body[0][1])
        elif action == 1:
            # Down
            next_location = (self.snake_body[0][0] + 1, self.snake_body[0][1])
        elif action == 2:
            # Left
            next_location = (self.snake_body[0][0], self.snake_body[0][1] - 1)
        elif action == 3:
            # Right
            next_location = (self.snake_body[0][0], self.snake_body[0][1] + 1)

        if self.check_game_over(next_location):
            return -1
        
        # Check if the next location is food
        if next_location == self.food_location:
            # Move the food to a new random location.
            food_location: tuple[int, int] = (
                torch.randint(0, self.board_size, (1,)).item(),
                torch.randint(0, self.board_size, (1,)).item(),
            )

            # Make sure the new food location is not on the snake body or next_location
            while food_location == next_location or food_location in self.snake_body:
                food_location = (
                    torch.randint(0, self.board_size, (1,)).item(),
                    torch.randint(0, self.board_size, (1,)).item(),
                )
            self.food_location = food_location
            self.board[self.food_location] = 1
            self.board[next_location] = -0.5
            self.board[self.snake_body[0]] = -1

            # Increament the snake length
            max_length = self.snake_body.maxlen + 1
            self.snake_body = deque(self.snake_body, maxlen=max_length)
            self.snake_body.appendleft(next_location)
            return 1
        
        # Move the snake
        self.board[next_location] = -0.5
        self.board[self.snake_body[0]] = -1
        self.board[self.snake_body[-1]] = 0
        self.snake_body.appendleft(next_location)
        return -0.01
    
    def check_game_over(self, next_location: tuple[int, int]) -> bool:
        # Begin by checking if the next location is out of bounds
        if next_location[0] < 0 or next_location[0] >= self.board_size or next_location[1] < 0 or next_location[1] >= self.board_size:
            return True
        
        # Check if the next location is snake body
        if next_location in self.snake_body:
            return True
        
        return False
            
    def print_board(self) -> None:
        # Print top border
        print("+" + "-" * (self.board_size * 2 + 1) + "+")
        
        for i in range(self.board_size):
            # Print left border
            print("| ", end="")
            
            for j in range(self.board_size):
                cell_value = self.board[i, j]
                
                # Different symbols for different elements:
                # 0 = empty space, -1 = snake body, -0.5 = snake head, 1 = food
                if cell_value == 0:
                    print("  ", end="")  # Empty space
                elif cell_value == -0.5 or cell_value == -1:
                    # Snake head is the first element in snake_location
                    if tuple([i, j]) == self.snake_body[0]:
                        print("■ ", end="")  # Snake head as solid block
                    else:
                        print("# ", end="")  # Snake body as shaded block
                elif cell_value == 1:
                    print("● ", end="")  # Food as circle
                else:
                    print("  ", end="")  # Default empty space
            
            # Print right border
            print(" |")
        
        # Print bottom border
        print("+" + "-" * (self.board_size * 2 + 1) + "+")

    def get_board(self) -> torch.Tensor:
        return self.board