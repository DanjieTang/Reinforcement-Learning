import torch
import torch.nn as nn
import torch.optim as opt
from snake_game import SnakeGame
import random
from collections import deque
import sys

# Create a global game instance.
game = SnakeGame()
game.init_game()

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, tensor: torch.Tensor):
        tensor = self.classifier(tensor)
        return tensor
    
def get_action(model: nn.Module, state: torch.Tensor, current_game: int, total_games: int) -> int:
    """
    Get the action to take.

    :param model: The model to get the action from.
    :param state: The state of the game.
    :param current_game: The current game number.
    :param total_games: The total number of games.
    :return: An integer representing the action to take. 0: up, 1: down, 2: left, 3: right.
    """
    # Epsilon-greedy exploration strategy
    # Epsilon decreases as the number of games increases
    epsilon = 1 * (total_games - current_game) / total_games
    state_on_device = state.to(next(model.parameters()).device) # Move state to model's device
    if random.random() < epsilon:
        return random.randint(0, 3) # Explore: random action
    else:
        # Exploit: choose the best action based on the model
        with torch.no_grad():
            action = model(state_on_device).argmax(dim=1)
            return action.item()
    
def train(model: nn.Module, optimizer: opt.Optimizer, loss_fn: nn.Module, memory: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]], num_games: int = 1000, sample_size: int = 5000):
    """
    Train the model for num_games games.

    :param model: The model to train.
    :param optimizer: The optimizer to use.
    :param loss_fn: The loss function to use.
    :param memory: Replay memory.
    :param num_games: The number of games to train the model on.
    :param sample_size: The sample size for training from memory.
    """
    history_scores = []
    device = next(model.parameters()).device # Get device from model
    for i in range(num_games):
        game.init_game()
        done = False
        score = 0
        idle_steps = 0 # Initialize idle_steps for each game

        while not done:
            current_state = game.get_compact_state().to(device)
            action = get_action(model, current_state, i, num_games)
            
            # Execute action and get reward
            reward_val = game.take_action(action)
            next_state_val: torch.Tensor = game.get_compact_state().to(device)

            if reward_val == 0.5: # Food eaten
                score += 1
                idle_steps = 0 # Reset idle steps on scoring
            elif reward_val == -0.5: # Game over
                done = True
            elif reward_val == -0.01: # Idle step
                idle_steps += 1

            if idle_steps >= 32: # Check for max idle steps
                done = True
                reward_val = -0.5 # Treat it as a game over condition

            # Store the transition in memory. This includes terminal states.
            memory.append((current_state, action, reward_val, next_state_val, done))

        # After the episode ends (done == True)
        if len(memory) > sample_size and i > 100: # Start training after 100 games and enough samples
            train_model(model, optimizer, loss_fn, memory, sample_size, device)
        
        history_scores.append(score)
        if (i + 1) % 100 == 0 and i > 100: # Log progress
            if history_scores: # Avoid division by zero if no scores
                print(f"Game {i+1}, Avg. Score (last 100): {sum(history_scores) / len(history_scores):.2f}, Memory Size: {len(memory)}")
            else:
                print(f"Game {i+1}, No scores to average yet.")
            history_scores = []

def train_model(model: nn.Module, optimizer: opt.Optimizer, criterion: nn.Module, memory: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]], sample_size: int = 1000, device: torch.device = torch.device("cpu")):
    """
    Train the model on the memory.
    """
    optimizer.zero_grad()

    if len(memory) < sample_size:
        mini_sample = random.sample(memory, len(memory))
    else:
        mini_sample = random.sample(memory, sample_size)
    
    # Unzip including the 'dones' flag
    states, actions, rewards, next_states, dones = zip(*mini_sample)

    states = torch.cat(states).to(device)
    # Ensure actions are stored as a tensor of the correct type (long) for indexing
    actions = torch.tensor(actions, device=device, dtype=torch.long) 
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, device=device, dtype=torch.bool) # Convert dones to a boolean tensor

    # Get model predictions
    pred = model(states)
    target = pred.clone()
    
    # Get Q-values for next states: Q(s', a') for all a'
    with torch.no_grad():
        next_q_values = model(next_states).max(dim=1)[0]
    
    # Calculate the target Q-value:
    # For terminal states (done=True), target = reward
    # For non-terminal states (done=False), target = reward + gamma * max_a' Q(s', a')
    # The (~dones) tensor acts as a mask: 0 for terminal, 1 for non-terminal.
    target_q_values_for_actions_taken = rewards + 0.9 * next_q_values * (~dones)
    
    # Update only the Q-values for the actions that were actually taken
    target[torch.arange(len(target)), actions] = target_q_values_for_actions_taken

    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

if __name__ == "__main__":
    # Determine the device to use
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda") # Fallback for other GPUs if needed
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = Model(input_size=16, hidden_size=256)
    model.to(device)  # Move the model to the selected device
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Calculate and print the number of parameters
    print(f"Total trainable parameters: {total_params}")
    
    optimizer = opt.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    memory_size = 50000
    sample_size = 5000
    memory: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]] = deque(maxlen=memory_size)

    train(model, optimizer, criterion, memory, num_games=100000, sample_size=sample_size)

    # Save the model
    torch.save(model.state_dict(), "snake_model.pth")
    print("Model saved to snake_model.pth")