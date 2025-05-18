import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------- Utility Functions -----------------------

def generate_unique_nonogram(grid_size, batch_size, existing_solutions=set()):
    solutions = []
    while len(solutions) < batch_size:
        new_solutions = np.random.randint(2, size=(batch_size, grid_size, grid_size))
        for solution in new_solutions:
            solution_tuple = tuple(map(tuple, solution))
            if solution_tuple not in existing_solutions:
                solutions.append(solution)
                existing_solutions.add(solution_tuple)
            if len(solutions) == batch_size:
                break
    solutions = np.array(solutions)
    row_clues = [[list(map(len, ''.join(map(str, row)).split('0'))) for row in solution] for solution in solutions]
    col_clues = [[list(map(len, ''.join(map(str, col)).split('0'))) for col in solution.T] for solution in solutions]
    row_clues = [[[clue for clue in clues if clue > 0] or [0] for clues in row] for row in row_clues]
    col_clues = [[[clue for clue in clues if clue > 0] or [0] for clues in col] for col in col_clues]
    return solutions, row_clues, col_clues, existing_solutions


def pad_clues(clues, max_len):
    return [clue + [0] * (max_len - len(clue)) for clue in clues]

# ----------------------- Model Definition -----------------------

class ClueTransformer(nn.Module):
    def __init__(self, grid_size, clue_max_len, clue_dim, num_heads, num_layers, model_dim):
        super().__init__()
        self.grid_size = grid_size
        self.embedding = nn.Embedding(clue_dim + 1, model_dim)
        self.model_dim = model_dim
        self.positional_encoding = nn.Parameter(torch.randn(1, clue_max_len * grid_size, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, clues):
        batch_size, num_clues, clue_len = clues.size()
        clues = clues.view(batch_size, -1)
        embedded_clues = self.embedding(clues)
        max_len = embedded_clues.size(1)
        if self.positional_encoding.size(1) < max_len:
            self.positional_encoding = nn.Parameter(torch.randn(1, max_len, self.model_dim).to(embedded_clues.device))
        embedded_clues = embedded_clues + self.positional_encoding[:, :max_len, :]
        return self.transformer(embedded_clues)


class PolicyNetwork(nn.Module):
    def __init__(self, grid_size, clue_max_len, clue_dim):
        super().__init__()
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8 * grid_size * grid_size, 16)
        self.row_clue_transformer = ClueTransformer(grid_size, clue_max_len, clue_dim, num_heads=2, num_layers=1, model_dim=16)
        self.col_clue_transformer = ClueTransformer(grid_size, clue_max_len, clue_dim, num_heads=2, num_layers=1, model_dim=16)
        self.fc2 = nn.Linear(16 * 2 + 16, 32)
        self.fc3 = nn.Linear(32, grid_size * grid_size * 2)

    def forward(self, state, row_clues, col_clues):
        state = state.to(device)
        row_clues = row_clues.to(device)
        col_clues = col_clues.to(device)

        x = state.unsqueeze(1).float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * self.grid_size * self.grid_size)
        x = F.relu(self.fc1(x))

        row_clues = self.row_clue_transformer(row_clues).mean(dim=1)
        col_clues = self.col_clue_transformer(col_clues).mean(dim=1)
        clues = torch.cat((row_clues, col_clues), dim=1)

        x = torch.cat((x, clues), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.grid_size * self.grid_size, 2)


class NonogramAgent:
    def __init__(self, grid_size, clue_max_len, clue_dim):
        self.policy_net = PolicyNetwork(grid_size, clue_max_len, clue_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.grid_size = grid_size

    def select_actions(self, states, row_clues, col_clues):
        states = torch.tensor(states, dtype=torch.float32)
        row_clues = torch.tensor(row_clues, dtype=torch.long)
        col_clues = torch.tensor(col_clues, dtype=torch.long)
        logits = self.policy_net(states, row_clues, col_clues)
        action_probs = torch.softmax(logits.view(states.size(0), -1), dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        flat_actions = action_dist.sample()
        actions = []
        for flat_action in flat_actions:
            idx = flat_action.item()
            pos = idx // 2
            value = idx % 2
            row = pos // self.grid_size
            col = pos % self.grid_size
            actions.append((row, col, value))
        return actions


# ----------------------- Environment -----------------------

class NonogramEnvironment:
    def __init__(self, grid_size, batch_size, streak_cap=5):
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.streak_cap = streak_cap
        self.solution, self.row_clues, self.col_clues = generate_unique_nonogram(grid_size, batch_size)[0:3]
        self.state = np.full((batch_size, grid_size, grid_size), -1, dtype=int)
        self.steps = np.zeros(batch_size, dtype=int)
        self.chosen_cells = [set() for _ in range(batch_size)]
        self.correct_guesses = [set() for _ in range(batch_size)]
        self.unique_guesses_streak = np.zeros(batch_size, dtype=int)

    def reset_with_solutions(self, solutions, row_clues, col_clues):
        self.solution = solutions
        self.row_clues = row_clues
        self.col_clues = col_clues
        return self.reset()

    def reset(self):
        self.state = np.full((self.batch_size, self.grid_size, self.grid_size), -1, dtype=int)
        self.steps = np.zeros(self.batch_size, dtype=int)
        self.chosen_cells = [set() for _ in range(self.batch_size)]
        self.correct_guesses = [set() for _ in range(self.batch_size)]
        self.unique_guesses_streak = np.zeros(self.batch_size, dtype=int)
        return self.state, self.row_clues, self.col_clues

    def step(self, actions):
        rewards = np.zeros(self.batch_size, dtype=float)
        done = np.zeros(self.batch_size, dtype=bool)
        for i, action in enumerate(actions):
            row, col, value = action
            self.steps[i] += 1
            if (row, col) in self.chosen_cells[i]:
                rewards[i] = -5
                self.unique_guesses_streak[i] = 0
            else:
                self.chosen_cells[i].add((row, col))
                self.unique_guesses_streak[i] += 1
                rewards[i] = min(self.unique_guesses_streak[i], self.streak_cap)
                if self.solution[i, row, col] == value:
                    rewards[i] += 2
                    self.correct_guesses[i].add((row, col))
                else:
                    rewards[i] -= 2
                self.state[i, row, col] = self.solution[i, row, col]
                if all(self.state[i, row, c] != -1 for c in range(self.grid_size)) and \
                   all(self.state[i, row, c] == self.solution[i, row, c] for c in range(self.grid_size)):
                    rewards[i] += 10
                if all(self.state[i, r, col] != -1 for r in range(self.grid_size)) and \
                   all(self.state[i, r, col] == self.solution[i, r, col] for r in range(self.grid_size)):
                    rewards[i] += 10
                if all(self.state[i, r, c] != -1 for r in range(self.grid_size) for c in range(self.grid_size)) and \
                   all(self.state[i, r, c] == self.solution[i, r, c] for r in range(self.grid_size) for c in range(self.grid_size)):
                    rewards[i] += 100
            done[i] = self._check_done(i)
        return self.state, rewards, done

    def _check_done(self, index):
        return np.array_equal(self.state[index], self.solution[index]) or self.steps[index] >= self.grid_size ** 2


# ----------------------- Checkpoint Loading -----------------------

def load_checkpoint(agent, optimizer, directory='models'):
    checkpoints = sorted(glob.glob(os.path.join(directory, 'checkpoint_*.pth')), key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    if checkpoints:
        checkpoint = torch.load(checkpoints[0], map_location=device)
        agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['clue_max_len'], checkpoint['clue_dim']
    return None, None

# ----------------------- GUI Functions -----------------------

class NonogramGUI:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.cell_size = 40
        self.root = tk.Tk()
        self.root.title('Nonogram Solver')
        canvas_size = grid_size * self.cell_size
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size)
        self.canvas.pack()
        self.rects = [[None]*grid_size for _ in range(grid_size)]
        for i in range(grid_size):
            for j in range(grid_size):
                x0 = j * self.cell_size
                y0 = i * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rect = self.canvas.create_rectangle(x0, y0, x1, y1, fill='white')
                self.rects[i][j] = rect

    def update_board(self, board):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                val = board[i][j]
                color = 'white'
                if val == 1:
                    color = 'black'
                elif val == 0:
                    color = 'lightgrey'
                self.canvas.itemconfig(self.rects[i][j], fill=color)
        self.root.update()

    def mainloop(self):
        self.root.mainloop()

# ----------------------- Main Execution -----------------------

def run_gui_solver():
    grid_size = 5
    clue_max_len = 3
    clue_dim = grid_size
    agent = NonogramAgent(grid_size, clue_max_len, clue_dim)
    optimizer = agent.optimizer
    saved_clue_max_len, saved_clue_dim = load_checkpoint(agent, optimizer)
    if saved_clue_max_len is not None:
        clue_max_len = saved_clue_max_len
    if saved_clue_dim is not None:
        clue_dim = saved_clue_dim

    solutions, row_clues, col_clues, _ = generate_unique_nonogram(grid_size, 1)
    row_clues = [pad_clues(rc, clue_max_len) for rc in row_clues]
    col_clues = [pad_clues(cc, clue_max_len) for cc in col_clues]

    env = NonogramEnvironment(grid_size, 1)
    states, row_clues, col_clues = env.reset_with_solutions(solutions, row_clues, col_clues)

    gui = NonogramGUI(grid_size)
    gui.update_board(states[0])
    time.sleep(1)

    done = False
    while not done:
        actions = agent.select_actions(states, row_clues, col_clues)
        states, rewards, done_flags = env.step(actions)
        done = done_flags[0]
        gui.update_board(states[0])
        time.sleep(0.5)

    gui.update_board(states[0])
    gui.root.title('Puzzle Solved!')
    gui.mainloop()

if __name__ == '__main__':
    run_gui_solver()
