# Quick Start Guide - Loss Game Experiments

## TL;DR

The import path issues have been fixed. You can now run experiments!

## 1. Verify the Fix Works

```bash
cd /mnt/users/clin/workspace/tic-tac-toe
python3 test_imports.py
```

Expected output: All checkmarks (✓) showing successful imports.

## 2. Run a Quick Test

```bash
python3 run_loss_game_experiments.py --game LossGame1D --solvers MCTS --analyze
```

This will:
- Run only the MCTS solver on LossGame1D
- Save results to `Data/loss_game_comparison/`
- Generate plots in `plots/LossGame1D/`
- Display summary statistics

## 3. Run Full Experiment

```bash
python3 run_loss_game_experiments.py --game both --analyze
```

This runs all 4 solvers on both game environments. Takes longer but gives comprehensive comparison.

## 4. Via SLURM (for cluster computing)

```bash
# Edit run_experiment_slurm.sh if needed (change game, solvers, etc.)
sbatch run_experiment_slurm.sh

# Monitor output
tail -f loss_game_experiment_<jobid>.out
```

## What Was Fixed

### Fix 1: Import Path Issues
The error:
```
ModuleNotFoundError: No module named 'environment'
```

Has been fixed by adding path manipulation in:
- `tests/test_loss_game_solvers.py` (lines 8-11)
- `sanity_check.py` (lines 3-6)

These changes ensure Python can find the root-level modules regardless of how the scripts are executed.

### Fix 2: Terminal State Evaluation
The error:
```
ValueError: Baseline environment is not terminal; cannot compute regret
```

Has been fixed by adding `_get_best_terminal_path()` function that:
- Follows the most visited children through MCTS tree to terminal depth
- Returns complete paths (e.g., `(2, 1, 4)`) instead of shallow ones (e.g., `(2,)`)
- Ensures the Evaluator receives terminal states with computed payoffs

See `TERMINAL_PATH_FIX.md` for detailed explanation.

## Files You Can Run

| Script | Purpose | Run From |
|--------|---------|----------|
| `test_imports.py` | Verify imports work | Root |
| `test_terminal_path_fix.py` | Verify terminal path fix works | Root |
| `sanity_check.py` | Quick functionality test | Root |
| `run_loss_game_experiments.py` | Main experiment runner | Root |
| `tests/test_loss_game_solvers.py` | Direct test execution | Root or via path |
| `tests/analyze_loss_game_results.py` | Analyze saved results | Root or via path |

## Expected Output Locations

```
Data/loss_game_comparison/
├── LossGame1D_20260119_143022.pkl
└── LossGame3D_20260119_143530.pkl

plots/
├── LossGame1D/
│   ├── LossGame1D_payoff_curves.png
│   ├── LossGame1D_regret_curves.png
│   ├── LossGame1D_final_comparison.png
│   ├── LossGame1D_efficiency.png
│   └── LossGame1D_all_players.png
└── LossGame3D/
    └── ... (same structure)
```

## Troubleshooting

### Import Errors
If you still see import errors:
1. Make sure you're in the root directory: `cd /mnt/users/clin/workspace/tic-tac-toe`
2. Check Python version: `python3 --version` (should be 3.7+)
3. Verify files exist: `ls tests/test_loss_game_solvers.py`
4. Run: `python3 test_imports.py` to verify all imports work

### Terminal State Errors
If you see "Baseline environment is not terminal":
1. This was fixed - make sure you have the latest version
2. Run: `python3 test_terminal_path_fix.py` to verify the fix works
3. If tree is too shallow, increase `num_simulations` in configs

## Next Steps

1. Run the quick test to verify everything works
2. Examine the generated plots
3. Read the detailed documentation in `LOSS_GAME_EXPERIMENTS_README.md`
4. Modify configurations in `tests/test_loss_game_solvers.py` if needed
5. Run full experiments on both games

For more details, see:
- `LOSS_GAME_EXPERIMENTS_README.md` - Comprehensive documentation
- `IMPORT_FIX_SUMMARY.md` - Details about the fix
