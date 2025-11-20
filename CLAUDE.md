# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research codebase for Monte Carlo Tree Search with Continuous actions (MCTSc), focusing on quantum game theory applications. The project implements several variants of MCTS algorithms including standard MCTS, Variational Gradient MCTS (VG-MCTS), and Differential Backward Induction (DBI) methods.

## Key Components

### Core Implementation (`old/src/`)
- **cmcts.py** - Main MCTS implementation with continuous action spaces and gradient-based updates
- **games.py** - Game definitions including extensive form games, quantum tic-tac-toe, and abstract game nodes
- **player.py** - Agent implementations for exact and quantum players
- **DBI.py** - Differential Backward Induction solver
- **IBR.py** - Iterated Best Response implementation

### Game Types
- **Extensive Form Games**: RandGame, Rand3DGame, Stackelberg
- **Board Games**: SimpleBoard (1D tic-tac-toe), TicTacToeBoard, QuantumTicTacToe
- **State Spaces**: StateSpace (quantum), clStateSpace (classical), QTTTStateSpace

### Action Spaces
- **CubeGrid**: Grid-based sampling in hypercube
- **BoxActionSpace**: 1D interval sampling  
- **SphereActionSpace**: Sampling on unit sphere surface using Gaussian normalization

## Common Commands

### Running Experiments
```bash
# Single experiment with quantum tic-tac-toe
python old/src/exec_cmcts_qt.py --lr 2e-4 --grad_freq 0.15 --grad_mode 1 --num_simu 100 --seed 42

# Generate configuration file for batch runs
python old/src/exec_cmcts_qt.py  # Edit create_config_file parameters first

# Run batch experiments (requires SLURM/MPI setup)
sbatch old/src/python.sh  # or multirun.sh
```

### Testing
```bash
# Run individual test files
python old/test/test_cmcts.py
python old/test/test_games.py
python old/test/test_dbi.py
```

### Interactive Development
```bash
# Launch Jupyter notebooks for analysis
jupyter notebook old/src/analysis.ipynb
jupyter notebook old/src/boardgame.ipynb
jupyter notebook old/src/plotdat.ipynb
```

## Algorithm Variants

The codebase implements three main MCTS variants controlled by gradient parameters:

1. **Standard MCTS** (`gradient_update_frequency=0`): Pure tree search without gradient updates
2. **Vanilla VG-MCTS** (`gradient_mode="vanilla"`): Local gradient-based action updates
3. **DBI VG-MCTS** (`gradient_mode="dbi"`): Uses Differential Backward Induction for gradient computation

## Data Structure

Experimental results are stored in `Data/quantum_tic_tac_toe/` as pickle files with naming convention:
`nsimu{num_simulations}_MCTS-{mode}-freq_{frequency}_lr_{learning_rate}{timestamp}.pkl`

## Key Classes

- **cMCTS**: Main MCTS algorithm with progressive widening and gradient updates
- **AbstractGameNode**: Tree node representation with abstract state tuples
- **GameRealization**: Maps abstract states to concrete game implementations
- **GameMetaData**: Defines branching ratios and game depth

## Development Notes

- Uses PyTorch for gradient computation and automatic differentiation
- Supports both CPU and distributed computing via MPI
- Random seeds are set consistently (default: 42) for reproducibility
- Progressive widening uses scaling function: `2 * x^(1/2)`
- Default exploration weight for UCB1: 1.0