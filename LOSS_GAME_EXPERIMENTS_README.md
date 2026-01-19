# Loss Game Solver Comparison Experiments

This framework provides a comprehensive comparison of different game solver algorithms (MCTS variants and local update methods) on both 1D and 3D loss game environments.

## Overview

The implementation consists of three main components:

1. **tests/test_loss_game_solvers.py** - Core test implementation with solver runners
2. **tests/analyze_loss_game_results.py** - Analysis and visualization tools
3. **run_loss_game_experiments.py** - High-level experiment runner

## Quick Start

### Running All Experiments

```bash
# Run all solvers on both games and generate analysis
python run_loss_game_experiments.py --game both --analyze

# Run with custom seed
python run_loss_game_experiments.py --game both --analyze --seed 123
```

### Running Specific Experiments

```bash
# Test only LossGame1D
python run_loss_game_experiments.py --game LossGame1D --analyze

# Test only specific solvers
python run_loss_game_experiments.py --game LossGame1D --solvers MCTS GradientMCTS --analyze

# Test without displaying plots (save only)
python run_loss_game_experiments.py --game both --analyze --no-display
```

### Analyzing Existing Results

```bash
# Analyze previously saved results
python run_loss_game_experiments.py --analyze-only Data/loss_game_comparison/LossGame1D_20260119_143022.pkl

# Or use the analysis script directly
python tests/analyze_loss_game_results.py Data/loss_game_comparison/LossGame1D_20260119_143022.pkl
```

## Solvers Compared

### 1. Standard MCTS
- Baseline tree search without optimization
- Pure exploration and exploitation via UCB
- Progressive simulation counts: 100, 200, 400, 800, 1600

### 2. GradientMCTS (VG-MCTS)
- MCTS with gradient-based action refinement
- Uses backpropagation through rollouts to improve actions
- Same simulation checkpoints as standard MCTS

### 3. MCTSwithLocalUpdate
- MCTS with local stochastic minimax updates
- Applies local optimization at each node during tree search
- Same simulation checkpoints as standard MCTS

### 4. LocalStochasticMinimaxUpdate
- Standalone local optimization without tree search
- Iteratively refines actions from a random starting trajectory
- Runs for 2000 iterations with evaluation every 100 iterations

## Game Environments

### LossGame1D
- 3 players, 3 stages
- Scalar (1D) actions per player
- Action bounds: [-5.0, 5.0]
- Complex coupled payoff functions

### LossGame3D
- 3 players, 3 stages
- 3D vector actions per player
- Action bounds: [-5.0, 5.0] per dimension
- More complex payoff structure

## Metrics Tracked

1. **Payoff**: Terminal payoff for each player
2. **Regret**: Gap to optimal via discretized minimax (lower is better)
3. **Convergence**: Payoff/regret improvement over iterations
4. **Efficiency**: Wall-clock time vs performance

## Configuration

You can modify the configurations in `tests/test_loss_game_solvers.py`:

```python
# Adjust solver checkpoints
SOLVER_CONFIGS["MCTS"]["simulation_checkpoints"] = [50, 100, 200]

# Adjust local update iterations
SOLVER_CONFIGS["LocalStochasticMinimaxUpdate"]["max_iter"] = 1000

# Adjust learning rates
SOLVER_CONFIGS["GradientMCTS"]["learning_rate"] = 5e-4

# Adjust evaluation branching factor
EVAL_CONFIG["branching_factor"] = 20
```

## Output Structure

### Results Files
Results are saved to `Data/loss_game_comparison/` with format:
```
Data/loss_game_comparison/
├── LossGame1D_20260119_143022.pkl
└── LossGame3D_20260119_143022.pkl
```

Each pickle file contains:
- `game_name`: Name of the game
- `timestamp`: When the experiment was run
- `results`: Dictionary mapping solver names to result lists
- `game_config`: Game configuration used
- `solver_configs`: Solver configurations used
- `eval_config`: Evaluation configuration used

### Plots
Plots are saved to `plots/<game_name>/` directory:
```
plots/
├── LossGame1D/
│   ├── LossGame1D_payoff_curves.png
│   ├── LossGame1D_regret_curves.png
│   ├── LossGame1D_final_comparison.png
│   ├── LossGame1D_efficiency.png
│   └── LossGame1D_all_players.png
└── LossGame3D/
    └── ... (same files)
```

## Testing the Implementation

### Sanity Check
A simple sanity check script is provided:
```bash
python sanity_check.py
```

This verifies:
- All imports work correctly
- Environments can be created
- Action spaces are configured properly
- MCTS can run successfully

### Small-Scale Test
To test with reduced computational budget (faster):

1. Edit `tests/test_loss_game_solvers.py`:
```python
# Temporarily change these values
SOLVER_CONFIGS["MCTS"]["simulation_checkpoints"] = [50, 100]
SOLVER_CONFIGS["LocalStochasticMinimaxUpdate"]["max_iter"] = 200
SOLVER_CONFIGS["LocalStochasticMinimaxUpdate"]["eval_interval"] = 50
```

2. Run:
```bash
python run_loss_game_experiments.py --game LossGame1D --analyze
```

3. Revert the changes after testing

### Individual Solver Testing
You can also test individual solvers directly:
```bash
# Test only MCTS on LossGame1D
python tests/test_loss_game_solvers.py --game LossGame1D --solvers MCTS

# Test multiple solvers
python tests/test_loss_game_solvers.py --game LossGame1D --solvers MCTS GradientMCTS
```

## Understanding the Results

### Expected Behavior

1. **Standard MCTS**: Should find reasonable solutions through exploration, but actions may not be locally optimal

2. **GradientMCTS**: Should show improved convergence due to gradient-based action refinement along explored paths

3. **MCTSwithLocalUpdate**: Should show better local optimization of actions via minimax sampling

4. **LocalStochasticMinimaxUpdate**: May converge quickly initially but could plateau without broader exploration

### Interpreting Plots

- **Payoff Curves**: Higher is generally better (depends on game structure)
- **Regret Curves**: Lower is always better (measures gap to optimal)
- **Final Comparison**: Bar charts comparing final performance
- **Efficiency Plots**: Shows which methods achieve good performance faster
- **All Players Plot**: Shows how each player's regret evolves

### Key Questions to Answer

1. Which solver finds the best solution (lowest regret)?
2. Which solver is most sample-efficient (best performance per simulation/iteration)?
3. Which solver is most time-efficient (best performance per second)?
4. How do gradient-based methods compare to sampling-based methods?
5. Is tree search (MCTS) necessary, or does local optimization suffice?

## Physics Background Notes

Since you have a physics background, here are some conceptual mappings:

- **MCTS**: Like Monte Carlo sampling in statistical mechanics - explores configuration space stochastically
- **Gradient Updates**: Like steepest descent or gradient flow in optimization
- **Local Minimax**: Like finding local energy minima in a potential landscape
- **Regret**: Similar to the gap between current state and ground state energy
- **Progressive Widening**: Adaptive discretization - refines action space where needed
- **UCB Exploration**: Balances exploration vs exploitation (analogous to temperature in simulated annealing)

## Troubleshooting

### Import Errors
If you get import errors, ensure you're running from the repository root:
```bash
cd /mnt/users/clin/workspace/tic-tac-toe
python run_loss_game_experiments.py --game LossGame1D --analyze
```

### Memory Issues
If you run out of memory with large MCTS trees:
- Reduce `simulation_checkpoints` to smaller values
- Test on LossGame1D first (simpler than 3D)

### Slow Execution
- Start with `--game LossGame1D --solvers MCTS` for quick testing
- Use `--no-display` to skip plot rendering during experiments

### Plot Display Issues
If plots don't display:
- Use `--no-display` flag and check saved PNG files
- Make sure matplotlib backend is configured correctly

## Next Steps

1. Run the basic experiment: `python run_loss_game_experiments.py --game LossGame1D --analyze`
2. Examine the generated plots in `plots/LossGame1D/`
3. Review the printed summary statistics
4. Compare results across different solvers
5. Optionally, modify configurations and rerun experiments
6. Scale up to both games: `python run_loss_game_experiments.py --game both --analyze`

## Questions and Clarifications

Feel free to:
- Modify hyperparameters in `SOLVER_CONFIGS`
- Add new solver variants
- Adjust evaluation metrics
- Create custom analysis plots

The framework is designed to be modular and extensible!
