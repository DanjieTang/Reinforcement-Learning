import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame
from torch.optim.lr_scheduler import CosineAnnealingLR

# Determine the device to use
if torch.backends.mps.is_available(): # Apple Silicon 
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
memory_size = 100000
sample_size = 1000
num_games = 10000
beginning_lr = 1e-3
ending_lr = 1e-5
max_step = 32

# Epsilon-greedy strategy parameters
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_DURATION_GAMES = int(num_games * 0.7)

class Agent(nn.Module):
    def __init__(self, board_size: int = 16, channel_size: list[int] = [1, 32, 64], hidden_size: int = 128):
        super(Agent, self).__init__()

        self.feature_extractor = nn.Sequential()

        for i in range(len(channel_size) - 1):
            self.feature_extractor.append(nn.Conv2d(channel_size[i], channel_size[i + 1], kernel_size=3, stride=1, padding=1))
            self.feature_extractor.append(nn.ReLU())
            self.feature_extractor.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.classifier = nn.Sequential(
            nn.Linear(channel_size[-1] * (board_size // 4) * (board_size // 4), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.size(0)
        x = self.feature_extractor(state)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        return x
    
def get_action(agent: Agent, state: torch.Tensor, current_game: int) -> int:
    """
    Get the action to take from the agent or randomly choose an action
    using an exponential decay epsilon-greedy strategy.

    :param agent: The agent to get the action from
    :param state: The current state of the game
    :param current_game: The current game number, used for epsilon calculation
    :return: The action to take
    """
    if current_game < EPSILON_DECAY_DURATION_GAMES:
        # Exponential decay
        # decay_factor should make epsilon reach EPSILON_END at EPSILON_DECAY_DURATION_GAMES
        # EPSILON_END = EPSILON_START * (decay_factor ** EPSILON_DECAY_DURATION_GAMES)
        # decay_factor = (EPSILON_END / EPSILON_START) ** (1 / EPSILON_DECAY_DURATION_GAMES)
        # To avoid issues if EPSILON_START is 0 or EPSILON_DECAY_DURATION_GAMES is 0 (though unlikely with current setup)
        if EPSILON_START == 0 or EPSILON_DECAY_DURATION_GAMES == 0:
             epsilon = EPSILON_END
        elif EPSILON_END >= EPSILON_START: # No decay needed if end is not smaller
             epsilon = EPSILON_START
        else:
            decay_factor = (EPSILON_END / EPSILON_START) ** (1 / EPSILON_DECAY_DURATION_GAMES)
            epsilon = EPSILON_START * (decay_factor ** current_game)
            epsilon = max(EPSILON_END, epsilon) # Ensure it doesn't go below end during decay
    else:
        epsilon = EPSILON_END

    if random.random() < epsilon:
        return random.randint(0, 3)  # Explore: choose a random action
    else:
        with torch.no_grad():
            # Add batch and channel dimension
            state = state.unsqueeze(0).unsqueeze(0).to(device)
            return agent.forward(state).argmax().item()
    
def train(agent: Agent, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler.CosineAnnealingLR, memory: deque[tuple[torch.Tensor, int, float, torch.Tensor]], criterion: nn.Module, num_games: int = 1000, batch_size: int = 1000):
    scores: list[int] = []

    for i in range(num_games):
        # Clear game status and initialize new game
        game.init_game()
        score: int = 0

        reward: float = 0
        step_counter: int = 0 # Make sure we don't get stuck in an infinite loop
        while reward != -1:
            # Get current state
            current_state: torch.Tensor = game.get_board()

            # Get action
            action: int = get_action(agent, current_state, i)
            reward: float = game.take_action(action)
            step_counter += 1
            if step_counter == max_step:
                reward = -1
            elif reward == 1:
                score += 1
                step_counter = 0

            # Get next state
            next_state: torch.Tensor = game.get_board()

            # Store transition
            memory.append((current_state, action, reward, next_state))

            train_short_memory(agent, optimizer, memory[-1], criterion)
        
        scores.append(score)
        train_long_memory(agent, optimizer, memory, criterion, batch_size)
        scheduler.step()

        if i % 100 == 0:
            print(f"Game {i}")
            print(f"Average score: {sum(scores) / len(scores)}")
            scores = []

def train_long_memory(agent: Agent, optimizer: optim.Optimizer, memory: deque[tuple[torch.Tensor, int, float, torch.Tensor]], criterion: nn.Module, batch_size: int = 1000):
    optimizer.zero_grad()
    
    if len(memory) < batch_size:
        mini_batch: list[tuple[torch.Tensor, int, float, torch.Tensor]] = random.sample(memory, len(memory))
    else:
        mini_batch: list[tuple[torch.Tensor, int, float, torch.Tensor]] = random.sample(memory, batch_size)
    
    states, actions, rewards, next_states = zip(*mini_batch)
    states = torch.stack(states).unsqueeze(1).to(device=device, dtype=torch.float32)
    actions = torch.tensor(actions, device=device, dtype=torch.long)
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_states = torch.stack(next_states).unsqueeze(1).to(device=device, dtype=torch.float32)
    
    # Get model predictions
    pred = agent(states)
    target = pred.clone()
    
    # Get Q-values for next states
    with torch.no_grad():
        next_q_values = agent(next_states).max(dim=1)[0]

    target_q_values_for_actions_taken = rewards + 0.9 * next_q_values * (~(rewards == -1))

    target[torch.arange(len(target)), actions] = target_q_values_for_actions_taken

    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()    

def train_short_memory(agent: Agent, optimizer: optim.Optimizer, memory: tuple[torch.Tensor, int, float, torch.Tensor], criterion: nn.Module):
    state, action, reward, next_state = memory
    # Add batch and channel dimension
    state = state.unsqueeze(0).unsqueeze(0).to(device)
    next_state = next_state.unsqueeze(0).unsqueeze(0).to(device)

    optimizer.zero_grad()

    pred = agent(state)
    target = pred.clone()

    with torch.no_grad():
        target[0, action] = reward + 0.9 * agent(next_state).max() * (~(reward == -1))

    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    
if __name__ == "__main__":
    board_size = 16
    game = SnakeGame(board_size=board_size)
    game.init_game()

    agent: Agent = Agent(board_size=board_size).to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"Total number of parameters in the model: {total_params}")
    optimizer: optim.Optimizer = optim.Adam(agent.parameters(), lr=beginning_lr)
    scheduler: optim.lr_scheduler.CosineAnnealingLR = CosineAnnealingLR(optimizer, T_max=num_games, eta_min=ending_lr)
    memory: deque[tuple[torch.Tensor, int, float, torch.Tensor]] = deque(maxlen=memory_size)
    criterion: nn.Module = nn.MSELoss()
    train(agent, optimizer, scheduler, memory, criterion, num_games=num_games, batch_size=sample_size)