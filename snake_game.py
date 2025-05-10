import random
import torch
from collections import deque
from typing import List

class SnakeGame():
    def __init__(self, width=16, height=16):
        self.width = width
        self.height = height
        self.snake_location = deque()
        self.board = torch.zeros((self.height, self.width), requires_grad=False)
        self.current_direction = 0 # 0: up, 1: down, 2: left, 3: right

    def init_game(self):
        # Clear the game status
        self.snake_location.clear()
        self.board = torch.zeros((self.height, self.width), requires_grad=False)
        # 0: up, 1: down, 2: left, 3: right
        initial_move_direction = random.randint(0,3)
        self.current_direction = initial_move_direction

        # Initialize the snake location in the middle section(25% to 75%) of the board
        head_location = [random.randint(self.width//4-1, self.width//4*3), random.randint(self.height//4-1, self.height//4*3)]
        self.snake_location.append(head_location)
        self.board[head_location[0], head_location[1]] = -0.5

        # Initialize the snake body
        direction = random.choice(['up', 'down', 'left', 'right'])
        if direction == 'up':
            self.snake_location.append([head_location[0]-1, head_location[1]])
        elif direction == 'down':
            self.snake_location.append([head_location[0]+1, head_location[1]])
        elif direction == 'left':
            self.snake_location.append([head_location[0], head_location[1]-1])
        elif direction == 'right':
            self.snake_location.append([head_location[0], head_location[1]+1])
        self.board[self.snake_location[1][0], self.snake_location[1][1]] = -1

        self.generate_food()

    def take_action(self, action: int) -> float:
        reward: float = 0
        if action == 0:
            reward = self.up()
        elif action == 1:
            reward = self.down()
        elif action == 2:
            reward = self.left()
        elif action == 3:
            reward = self.right()
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Update current_direction based on action.
        # This assumes the action directly corresponds to a new direction if valid.
        # If the game logic implies direction differently (e.g. action is relative turn),
        # this part might need adjustment.
        # For now, let's assume action is absolute direction if the move is valid.
        # We need to check if the move was valid before updating direction.
        # The move method returns -0.5 on game over (invalid move).
        if reward != -0.5: # if move was valid
            self.current_direction = action
        
        return reward

    def up(self) -> float:
        new_head_location = [self.snake_location[0][0]-1, self.snake_location[0][1]]
        return self.move(new_head_location)

    def down(self) -> float:
        new_head_location = [self.snake_location[0][0]+1, self.snake_location[0][1]]
        return self.move(new_head_location)

    def left(self) -> float:
        new_head_location = [self.snake_location[0][0], self.snake_location[0][1]-1]
        return self.move(new_head_location)

    def right(self) -> float:
        new_head_location = [self.snake_location[0][0], self.snake_location[0][1]+1]
        return self.move(new_head_location)

    def move(self, new_head_location: List[int]):
        # Move the snake head
        if self.check_game_over(new_head_location):
            return -0.5

        # Move the snake tail
        is_food = self.check_food(new_head_location)
        if not is_food:
            tail_location = self.snake_location.pop()
            self.board[tail_location[0], tail_location[1]] = 0
      
        self.snake_location.appendleft(new_head_location)
        # Set new head
        self.board[self.snake_location[0][0], self.snake_location[0][1]] = -0.5
        # Set previous head to body
        if len(self.snake_location) > 1:
            self.board[self.snake_location[1][0], self.snake_location[1][1]] = -1

        if is_food:
            self.generate_food()
            return 0.5

        return -0.01

    def generate_food(self):
        # Generate a random food location
        while True:
            food_location = [random.randint(0, self.height-1), random.randint(0, self.width-1)]
            if self.board[food_location[0], food_location[1]] == 0:
                break
        self.board[food_location[0], food_location[1]] = 1

    def check_game_over(self, new_head_location: List[int]):
        if new_head_location[0] < 0 or new_head_location[0] >= self.height or new_head_location[1] < 0 or new_head_location[1] >= self.width:
            return True
        if self.board[new_head_location[0], new_head_location[1]] == -1:
            return True
        return False
    
    def check_food(self, new_head_location: List[int]):
        if self.board[new_head_location[0], new_head_location[1]] == 1:
            return True
        return False

    def print_board(self):
        # Clear terminal for better visualization
        # print("\033[H\033[J", end="")
        
        # Print top border
        print("+" + "-" * (self.width * 2 + 1) + "+")
        
        for i in range(self.height):
            # Print left border
            print("| ", end="")
            
            for j in range(self.width):
                cell_value = self.board[i, j]
                
                # Different symbols for different elements:
                # 0 = empty space, -1 = snake body, 1 = food
                if cell_value == 0:
                    print("  ", end="")  # Empty space
                elif cell_value == -1:
                    # Snake head is the first element in snake_location
                    if [i, j] == self.snake_location[0]:
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
        print("+" + "-" * (self.width * 2 + 1) + "+")

    def view_board(self):
        return self.board.unsqueeze(0).unsqueeze(0).clone()

    def get_compact_state(self) -> torch.Tensor:
        head_x, head_y = self.snake_location[0]

        # Wall distances (normalized)
        dist_wall_up = head_x / self.height
        dist_wall_down = (self.height - 1 - head_x) / self.height
        dist_wall_left = head_y / self.width
        dist_wall_right = (self.width - 1 - head_y) / self.width
        wall_distances = torch.tensor([dist_wall_up, dist_wall_down, dist_wall_left, dist_wall_right], dtype=torch.float32)

        # Food presence (binary)
        food_up = 1.0 if head_x > 0 and self.board[head_x - 1, head_y] == 1 else 0.0
        food_down = 1.0 if head_x < self.height - 1 and self.board[head_x + 1, head_y] == 1 else 0.0
        food_left = 1.0 if head_y > 0 and self.board[head_x, head_y - 1] == 1 else 0.0
        food_right = 1.0 if head_y < self.width - 1 and self.board[head_x, head_y + 1] == 1 else 0.0
        food_presence = torch.tensor([food_up, food_down, food_left, food_right], dtype=torch.float32)

        # Body presence (binary)
        # Check immediate N, S, E, W cells for snake body parts (-1)
        body_up = 1.0 if head_x > 0 and self.board[head_x - 1, head_y] == -1 else 0.0
        body_down = 1.0 if head_x < self.height - 1 and self.board[head_x + 1, head_y] == -1 else 0.0
        body_left = 1.0 if head_y > 0 and self.board[head_x, head_y - 1] == -1 else 0.0
        body_right = 1.0 if head_y < self.width - 1 and self.board[head_x, head_y + 1] == -1 else 0.0
        body_presence = torch.tensor([body_up, body_down, body_left, body_right], dtype=torch.float32)

        # Current direction (one-hot encoded)
        # 0: up, 1: down, 2: left, 3: right
        direction_vector = torch.zeros(4, dtype=torch.float32)
        if self.snake_location: # Ensure snake exists
             # current_direction should be updated when an action is successfully taken
            direction_vector[self.current_direction] = 1.0

        compact_state = torch.cat((wall_distances, food_presence, body_presence, direction_vector)).unsqueeze(0)
        return compact_state