#!/usr/bin/env python
"""Nonogram Solver – sleek dark-theme GUI powered by Tkinter + ttkbootstrap.
   ░ Modern "Superhero" theme (dark, accessible colours)
   ░ Side panel now includes a striped progress-bar & legend for quick orientation
   ░ Solution view is collapsible so newcomers focus on the puzzle first
   ░ Internals (agent, environment) unchanged → solver logic identical
"""

import glob
import logging
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tkinter as tk
import tkinter.ttk as ttk
from ttkbootstrap import Style

# ───────────────────── Logging setup ──────────────────────
_loglevel = os.getenv("LOGLEVEL", "INFO").upper()
logging.basicConfig(
    level=_loglevel,
    format="[%(levelname)s] %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------
#  Device (CPU ↔ CUDA)
# ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Running on {device} · PyTorch {torch.__version__}")

# ---------------------------------------------------------------
#  Utility helpers – puzzle generation / clue padding
# ---------------------------------------------------------------

def generate_unique_nonogram(grid_size: int, batch_size: int, existing_solutions=set()):
    """Return batch of *unique* puzzles plus their row/col clues."""
    solutions = []
    while len(solutions) < batch_size:
        new = np.random.randint(2, size=(batch_size, grid_size, grid_size))
        for sol in new:
            tup = tuple(map(tuple, sol))
            if tup not in existing_solutions:
                solutions.append(sol)
                existing_solutions.add(tup)
            if len(solutions) == batch_size:
                break
    solutions = np.asarray(solutions)

    row_clues, col_clues = [], []
    for sol in solutions:
        row_clues.append([[len(s) for s in "".join(map(str, r)).split("0") if s] or [0] for r in sol])
        col_clues.append([[len(s) for s in "".join(map(str, c)).split("0") if s] or [0] for c in sol.T])
    return solutions, row_clues, col_clues, existing_solutions


def pad_clue(clue: list[int], max_len: int):
    """Pad each clue row / column out to a fixed width for the Transformer."""
    return clue + [0] * (max_len - len(clue))


# ---------------------------------------------------------------
#  Model definition – identical to previous version
# ---------------------------------------------------------------

class ClueTransformer(nn.Module):
    def __init__(self, grid: int, max_len: int, vocab: int, heads: int, layers: int, dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab + 1, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len * grid, dim))
        block = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.xf = nn.TransformerEncoder(block, layers)

    def forward(self, x):  # (B, grid, max_len)
        b, g, l = x.shape
        x = x.view(b, -1)
        e = self.embed(x)
        # positional enc up-/down-sample if required
        if self.pos_enc.size(1) < e.size(1):
            self.pos_enc = nn.Parameter(torch.randn(1, e.size(1), e.size(-1), device=e.device))
        e = e + self.pos_enc[:, : e.size(1)]
        return self.xf(e)


class PolicyNetwork(nn.Module):
    def __init__(self, grid: int, max_len: int, vocab: int):
        super().__init__()
        self.grid = grid
        # board branch (conv->FC)
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.fc1 = nn.Linear(8 * grid * grid, 16)
        # clue branches
        self.row_trans = ClueTransformer(grid, max_len, vocab, heads=2, layers=1, dim=16)
        self.col_trans = ClueTransformer(grid, max_len, vocab, heads=2, layers=1, dim=16)
        # fusion & output
        self.fc2 = nn.Linear(16 * 3, 32)
        self.fc3 = nn.Linear(32, grid * grid * 2)  # two choices per cell (fill / empty)

    def forward(self, s, r, c):
        s = s.to(device)
        r = r.to(device)
        c = c.to(device)

        x = F.relu(self.conv1(s.unsqueeze(1).float()))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 8 * self.grid * self.grid)
        x = F.relu(self.fc1(x))

        r = self.row_trans(r).mean(1)
        c = self.col_trans(c).mean(1)
        x = torch.cat((x, r, c), dim=1)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, self.grid * self.grid, 2)


class NonogramAgent:
    def __init__(self, grid: int, max_len: int, vocab: int):
        self.net = PolicyNetwork(grid, max_len, vocab).to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=1e-3)
        self.grid = grid

    def select_actions(self, state, row, col):
        """Sample one action per board cell from the learned policy."""
        state_t = torch.tensor(state, dtype=torch.float32)
        row_t = torch.tensor(row, dtype=torch.long)
        col_t = torch.tensor(col, dtype=torch.long)
        logits = self.net(state_t, row_t, col_t)
        probs = torch.softmax(logits.view(state_t.size(0), -1), dim=-1)
        dist = torch.distributions.Categorical(probs)
        flat = dist.sample()

        actions = []
        for idx in flat:
            idx = idx.item()
            pos = idx // 2
            val = idx % 2
            actions.append((pos // self.grid, pos % self.grid, val))
        return actions


# ---------------------------------------------------------------
#  Nonogram environment (unchanged)
# ---------------------------------------------------------------

class NonogramEnvironment:
    def __init__(self, grid: int, batch: int, streak_cap: int = 5):
        self.grid = grid
        self.batch = batch
        self.cap = streak_cap
        self.solution, self.row_clues, self.col_clues, _ = generate_unique_nonogram(grid, batch)
        self.reset()

    def reset_with_solutions(self, sol, row, col):
        self.solution, self.row_clues, self.col_clues = sol, row, col
        return self.reset()

    def reset(self):
        self.state = np.full((self.batch, self.grid, self.grid), -1)
        self.steps = np.zeros(self.batch, int)
        self.chosen = [set() for _ in range(self.batch)]
        self.streak = np.zeros(self.batch, int)
        return self.state, self.row_clues, self.col_clues

    def _done(self, i):
        solved = np.array_equal(self.state[i], self.solution[i])
        step_limit = self.steps[i] >= self.grid ** 2
        return solved or step_limit

    def step(self, acts):
        reward = np.zeros(self.batch)
        done = np.zeros(self.batch, bool)

        for i, (row, col, val) in enumerate(acts):
            self.steps[i] += 1
            if (row, col) in self.chosen[i]:
                reward[i] = -5
                self.streak[i] = 0
            else:
                self.chosen[i].add((row, col))
                self.streak[i] += 1
                reward[i] = min(self.streak[i], self.cap)
                reward[i] += 2 if self.solution[i, row, col] == val else -2
                self.state[i, row, col] = self.solution[i, row, col]

                # bonuses for completing a row/column correctly
                if all(self.state[i, row, c] != -1 for c in range(self.grid)) and np.array_equal(self.state[i, row], self.solution[i, row]):
                    reward[i] += 10
                if all(self.state[i, r, col] != -1 for r in range(self.grid)) and np.array_equal(self.state[i, :, col], self.solution[i, :, col]):
                    reward[i] += 10
                if np.array_equal(self.state[i], self.solution[i]):
                    reward[i] += 100

            done[i] = self._done(i)
        return self.state, reward, done


# ---------------------------------------------------------------
#  Checkpoint helpers (weights are optional)
# ---------------------------------------------------------------

def load_checkpoint(agent, optim, directory="models"):
    ckpts = sorted(
        glob.glob(os.path.join(directory, "checkpoint_*.pth")),
        key=lambda p: int(p.split("_")[-1].split(".")[0]),
        reverse=True,
    )
    if not ckpts:
        log.warning("No checkpoints found.")
        return None, None
    path = ckpts[0]
    log.info(f"Loading {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    agent.net.load_state_dict(ckpt["model_state_dict"], strict=False)
    optim.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["clue_max_len"], ckpt["clue_dim"]


# ---------------------------------------------------------------
#  Sleek GUI (ttkbootstrap “Superhero”)
# ---------------------------------------------------------------

class NonogramGUI:
    """Encapsulates *all* Tk widgets & styling."""

    PAD = 16
    CLUE_SPACE = 72
    CELL = 44
    LEGEND_SIZE = 18

    def __init__(self, grid: int, row_clues, col_clues, solution):
        self.grid = grid
        self.rows = row_clues
        self.cols = col_clues
        self.solution = solution

        # ── Themed root ------------------------------------------------------
        style = Style("superhero")  # dark & saturated
        self.root: tk.Tk = style.master
        self.root.title("AI-Powered Nonogram Solver")
        self.root.minsize(700, 520)

        # palette shortcuts
        COL_BG = style.colors.dark
        COL_CARD = style.colors.secondary
        COL_ACCENT = style.colors.primary
        COL_GRID = style.colors.border
        COL_EMPTY = style.colors.light

        self._accent = COL_ACCENT

        # universal fonts
        style.configure("Heading.TLabel", font=("Inter", 14, "bold"))
        style.configure("Body.TLabel", font=("Inter", 10))

        # scaffolding (two cards: board + sidebar)
        main = ttk.Frame(self.root, padding=self.PAD)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        board_card = ttk.Frame(main, padding=self.PAD, style="surface.TFrame")
        side_card = ttk.Frame(main, padding=self.PAD, style="surface.TFrame")
        board_card.grid(row=0, column=0, sticky="nsew", padx=(0, self.PAD))
        side_card.grid(row=0, column=1, sticky="n")

        # ► Board title
        ttk.Label(board_card, text="Nonogram Puzzle", style="Heading.TLabel").pack(anchor="w", pady=(0, 8))

        # ► Active board canvas
        px = self.CLUE_SPACE + self.grid * self.CELL
        self.canvas = tk.Canvas(board_card, width=px, height=px, bg=COL_CARD, highlightthickness=1, highlightbackground=COL_GRID)
        self.canvas.pack()

        # ► Solution (toggleable)
        self.sol_shown = tk.BooleanVar(value=True)
        self.toggle_btn = ttk.Checkbutton(board_card, text="Solution", bootstyle=("primary", "toolbutton"), variable=self.sol_shown, command=self._toggle_solution)
        self.toggle_btn.pack(anchor="w", pady=(10, 4))

        self.sol_canvas = tk.Canvas(board_card, width=self.grid * self.CELL, height=self.grid * self.CELL, bg=COL_CARD, highlightthickness=1, highlightbackground=COL_GRID)

        # ► Canvas rectangles
        self.rects = [[None] * self.grid for _ in range(self.grid)]
        self.sol_rects = [[None] * self.grid for _ in range(self.grid)]

        for r in range(self.grid):
            for c in range(self.grid):
                # active board
                x0 = self.CLUE_SPACE + c * self.CELL
                y0 = self.CLUE_SPACE + r * self.CELL
                rect = self.canvas.create_rectangle(x0, y0, x0 + self.CELL, y0 + self.CELL, fill=COL_CARD, outline=COL_GRID)
                self.rects[r][c] = rect
                # solution board (pre-draw, hidden until toggled)
                xs = c * self.CELL
                ys = r * self.CELL
                col_fill = "#000000" if self.solution[r][c] == 1 else COL_EMPTY
                self.sol_rects[r][c] = self.sol_canvas.create_rectangle(xs, ys, xs + self.CELL, ys + self.CELL, fill=col_fill, outline=COL_GRID)

        # ► Clues
        def _fmt(nums, sep=" "):
            return sep.join(map(str, [n for n in nums if n])) or "0"

        for r in range(self.grid):
            self.canvas.create_text(self.CLUE_SPACE - 10, self.CLUE_SPACE + r * self.CELL + self.CELL / 2, text=_fmt(self.rows[r]), anchor="e", font=("Inter", 10, "bold"), fill=COL_ACCENT)
        for c in range(self.grid):
            self.canvas.create_text(self.CLUE_SPACE + c * self.CELL + self.CELL / 2, self.CLUE_SPACE - 10, text=_fmt(self.cols[c], "\n"), anchor="s", font=("Inter", 10, "bold"), fill=COL_ACCENT)

        # ── Sidebar -----------------------------------------------------------
        ttk.Label(side_card, text="How to Play", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(side_card, text=(
            "Fill rows & columns so their clue numbers match.\n"
            "- Numbers = length of consecutive black blocks.\n"
            "- Blocks are separated by 1+ empty cells.\n"), style="Body.TLabel", wraplength=210, justify="left").pack(anchor="w", pady=(4, 12))

        # ► Legend
        ttk.Label(side_card, text="Legend", style="Heading.TLabel").pack(anchor="w", pady=(0, 4))
        legend = ttk.Frame(side_card)
        legend.pack(anchor="w", pady=(0, 14))
        self._legend_square(legend, "#000000").grid(row=0, column=0)
        ttk.Label(legend, text="Filled", style="Body.TLabel").grid(row=0, column=1, padx=(6, 12))
        self._legend_square(legend, COL_EMPTY).grid(row=1, column=0)
        ttk.Label(legend, text="Empty", style="Body.TLabel").grid(row=1, column=1, padx=(6, 12))
        self._legend_square(legend, COL_CARD).grid(row=2, column=0)
        ttk.Label(legend, text="Unknown", style="Body.TLabel").grid(row=2, column=1, padx=(6, 0))

        # ► Progress metrics
        self.step_var = tk.StringVar(value="Steps: 0")
        self.status_var = tk.StringVar(value="Solving…")
        ttk.Label(side_card, textvariable=self.step_var, style="Body.TLabel").pack(anchor="w")
        self.progress = ttk.Progressbar(side_card, maximum=grid * grid, value=0, length=190, mode="determinate", bootstyle="info-striped")
        self.progress.pack(anchor="w", pady=(0, 10))
        ttk.Label(side_card, textvariable=self.status_var, style="Body.TLabel").pack(anchor="w", pady=(0, 12))

        # ► Selection log (Treeview)
        ttk.Label(side_card, text="Selection Log", style="Heading.TLabel").pack(anchor="w", pady=(0, 4))
        log_frame = ttk.Frame(side_card)
        log_frame.pack(anchor="w", fill="both", expand=True)
        cols = ("cell", "choice", "result")
        self.log_tv = ttk.Treeview(log_frame, columns=cols, show="headings", height=12, selectmode="none", bootstyle="dark")
        for cid, hdr, width in zip(cols, ("Cell", "Chosen", "Result"), (70, 70, 70)):
            self.log_tv.heading(cid, text=hdr)
            self.log_tv.column(cid, anchor="center", width=width, stretch=True)
        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_tv.yview, bootstyle="dark-round")
        self.log_tv.configure(yscrollcommand=vsb.set)
        self.log_tv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        # ► Buttons
        btn_frame = ttk.Frame(side_card)
        btn_frame.pack(anchor="w", pady=(12, 0))
        self.new_btn = ttk.Button(btn_frame, text="Reset Puzzle", bootstyle="secondary")
        self.new_btn.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btn_frame, text="Quit", bootstyle="danger", command=self.root.destroy).grid(row=0, column=1)

        # paint once
        self.root.update()

        # private state
        self._color_map = {1: "#000000", 0: COL_EMPTY, -1: COL_CARD}
        self._flash_map = {"Correct": "#8BC34A", "Incorrect": "#FF455E", "Duplicate": "#FFC107"}
        self._grid_outline = COL_GRID

    def _refresh_palette(self):
        """Re-derive the current palette from ttkbootstrap’s theme."""
        style          = Style()
        COL_CARD       = style.colors.secondary
        COL_EMPTY      = style.colors.light
        COL_GRID       = style.colors.border
        self._color_map = {1: "#000000", 0: COL_EMPTY, -1: COL_CARD}
        self._grid_outline = COL_GRID

    # ─── Helper widgets ─────────────────────────────────────────────
    def _legend_square(self, parent: ttk.Frame, color: str):
        c = tk.Canvas(parent, width=self.LEGEND_SIZE, height=self.LEGEND_SIZE, highlightthickness=0)
        c.create_rectangle(0, 0, self.LEGEND_SIZE, self.LEGEND_SIZE, fill=color, outline="")
        return c

    def _toggle_solution(self):
        if self.sol_shown.get():
            self.sol_canvas.pack(after=self.toggle_btn,
                                 padx=(self.CLUE_SPACE, 0),
                                 pady=(8, 0))     # ← gentle vertical spacing
        else:
            self.sol_canvas.pack_forget()

    # ─── Public API for Controller ─────────────────────────────────
    def update_board(self, board, step: int | None = None):
        for r in range(self.grid):
            for c in range(self.grid):
                self.canvas.itemconfig(self.rects[r][c], fill=self._color_map[board[r][c]])
        if step is not None:
            self.step_var.set(f"Steps: {step}")
            self.progress["value"] = min(step, self.progress["maximum"])
        self.root.update_idletasks()

    def flash_cell(self, r: int, c: int, verdict: str):
        cid = self.rects[r][c]
        self.canvas.itemconfig(cid, outline=self._flash_map[verdict], width=3)
        self.canvas.after(150, lambda: self.canvas.itemconfig(cid, outline=self._grid_outline, width=1))

    def log_selection(self, row: int, col: int, val: int, result: str):
        self.log_tv.insert("", "end", values=(f"({row+1},{col+1})", "Filled" if val == 1 else "Empty", result))
        self.log_tv.yview_moveto(1.0)

    def rebuild(self, rows, cols, solution):
        """Swap-in a brand-new puzzle (after Reset)."""
        self.rows, self.cols, self.solution = rows, cols, solution
        # clear canvases
        self.canvas.delete("all")
        self.sol_canvas.delete("all")
        # redraw rects + clues
        self.rects = [[None] * self.grid for _ in range(self.grid)]
        self.sol_rects = [[None] * self.grid for _ in range(self.grid)]
        for r in range(self.grid):
            for c in range(self.grid):
                x0 = self.CLUE_SPACE + c * self.CELL
                y0 = self.CLUE_SPACE + r * self.CELL
                self.rects[r][c] = self.canvas.create_rectangle(x0, y0, x0 + self.CELL, y0 + self.CELL, fill=self._color_map[-1], outline=self._grid_outline)
                xs, ys = c * self.CELL, r * self.CELL
                fill = "#000000" if solution[r][c] == 1 else self._color_map[0]
                self.sol_rects[r][c] = self.sol_canvas.create_rectangle(xs, ys, xs + self.CELL, ys + self.CELL, fill=fill, outline=self._grid_outline)
        _fmt = lambda nums, sep=" ": sep.join(map(str, [n for n in nums if n])) or "0"
        for r in range(self.grid):
            self.canvas.create_text(self.CLUE_SPACE - 10,
                                    self.CLUE_SPACE + r*self.CELL + self.CELL/2,
                                    text=_fmt(rows[r]),
                                    anchor="e",
                                    font=("Inter", 10, "bold"),
                                    fill=self._accent)
        for c in range(self.grid):
            self.canvas.create_text(self.CLUE_SPACE + c*self.CELL + self.CELL/2,
                                    self.CLUE_SPACE - 10,
                                    text=_fmt(cols[c], "\n"),
                                    anchor="s",
                                    font=("Inter", 10, "bold"),
                                    fill=self._accent)

    def mainloop(self):
        self.root.mainloop()


# ---------------------------------------------------------------
#  Controller – event-driven, no blocking sleep
# ---------------------------------------------------------------

def run_gui_solver():
    grid = 5
    clue_max_len = 3
    vocab = 5

    agent = NonogramAgent(grid, clue_max_len, vocab)
    load_checkpoint(agent, agent.opt)

    # shared mutable state
    env = None
    state = None
    row_pad = col_pad = None
    sols = row_c = col_c = None
    step = 0
    done = False
    guessed_cells: set[tuple[int, int]] = set()

    # build initial puzzle
    def build_puzzle():
        nonlocal env, state, row_pad, col_pad, sols, row_c, col_c, step, done, guessed_cells
        sols, row_c, col_c, _ = generate_unique_nonogram(grid, 1)
        row_disp = [pad_clue(c, clue_max_len) for c in row_c[0]]
        col_disp = [pad_clue(c, clue_max_len) for c in col_c[0]]
        env = NonogramEnvironment(grid, 1)
        state, _, _ = env.reset_with_solutions(sols, row_c, col_c)
        row_pad = np.array([row_disp])
        col_pad = np.array([col_disp])
        step = 0
        done = False
        guessed_cells.clear()
        return row_disp, col_disp

    row_disp, col_disp = build_puzzle()
    gui = NonogramGUI(grid, row_disp, col_disp, sols[0])

    def reset_puzzle():
        row_disp, col_disp = build_puzzle()
        gui.log_tv.delete(*gui.log_tv.get_children())
        gui.rebuild(row_disp, col_disp, sols[0])
        gui.step_var.set("Steps: 0")
        gui.progress["value"] = 0
        gui.status_var.set("Solving...")
        gui.update_board(state[0])
        gui.root.after(400, step_solver)

    gui.new_btn.configure(command=reset_puzzle)

    # solver tick --------------------------------------------------
    def step_solver():
        nonlocal state, step, done
        if done:
            return
        actions = agent.select_actions(state, row_pad, col_pad)
        state, _, flags = env.step(actions)

        all_duplicates = True
        for r, c, v in actions:
            if (r, c) in guessed_cells:
                verdict = "Duplicate"
            else:
                verdict = "Correct" if state[0, r, c] == sols[0, r, c] else "Incorrect"
                guessed_cells.add((r, c))
                all_duplicates = False
            gui.flash_cell(r, c, verdict)
            gui.log_selection(r, c, v, verdict)

        step += 1
        gui.update_board(state[0], step)
        done = flags[0]
        delay = 50 if all_duplicates else 400
        if not done:
            gui.root.after(delay, step_solver)
        else:
            gui.status_var.set("Puzzle Solved!")
            gui.progress["value"] = gui.progress["maximum"]

    gui.root.after(400, step_solver)
    gui.mainloop()


if __name__ == "__main__":
    run_gui_solver()
