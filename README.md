# Tetris Reinforcement Learning vs Heuristic Agent






## Project Structure

```
.
├── board.py               # Board implementation
├── game.py                # Tetris environment
├── block.py               # Tetromino definitions
├── heuristic.py           # Heuristic agent
├── rl_agent.py            # Deep Q-Network implementation
├── evaluate_rl.py         # RL evaluation/ used for rq3/4
├── evaluate.py            # Heuristic evaluation/ used for rq3/4
├── rq2.py                 # Reward experiments
├── rq3.py                 # Feature selection experiments
├── graph.py               # Visualization scripts
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
cd <repository-name>
```

Install the required packages

```bash
pip install -r requirements.txt
```

---

## Training the Reinforcement Learning Agent for each reward function

Run

```bash
python rq2.py
```

The trained model will be saved automatically.

---

## Evaluating a RL Agent

Run

```bash
python evaluate.py
```

The evaluation reports

- Average score
- Average lines cleared
- Average moves survived
- Average Tetrises
- Average holes created
- Average holes removed
- Average bomb usage (when enabled)

---

## Running Experiments



### Research Question 1

Evaluate different reward functions or state representations.

```bash
python rq2.py
```

### Research Question 2

Perform state representation experiments.

run
python rq3.py

### Research Question 3/4
Using tetris_rl_state_high.pt as weights and use_bombs=False for rq3 and True for rq4 
run 
python evaluate_rl.py 
python evaluate_heuristic.py 

---

