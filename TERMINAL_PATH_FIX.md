# Terminal Path Fix Documentation

## Problem

The original implementation had an error when evaluating MCTS results:

```
ValueError: Baseline environment is not terminal; cannot compute regret
```

This occurred in `evaluator.py` line 140 when trying to compute regret for a non-terminal state.

## Root Cause

### The Issue

1. **MCTS Structure**: MCTS builds a tree where nodes at different depths represent game states after different numbers of moves
   - Root `()`: Initial state (0 actions)
   - Depth 1 `(2,)`: State after 1 action
   - Depth 2 `(2, 1)`: State after 2 actions
   - Depth 3 `(2, 1, 4)`: State after 3 actions → **TERMINAL**

2. **What `select_action()` Returns**:
   ```python
   best_child, _ = mcts.select_action()
   # Returns: (2,)  - Only depth 1!
   ```
   - This method returns the **immediate best child** of the root
   - Designed for iterative play: pick best move, execute it, repeat
   - Returns a **shallow** node (depth 1)

3. **What We Need for Evaluation**:
   - A **complete terminal trajectory** (depth = `num_stages`)
   - For LossGame with 3 stages: path like `(2, 1, 4)` (3 actions)
   - Only terminal states have computed payoffs

4. **The Mismatch**:
   ```python
   best_child = (2,)  # Depth 1, non-terminal
   terminal_env = reconstruct(env, mcts, best_child)
   # terminal_env.terminal = False
   # terminal_env.payoff = None

   evaluator = Evaluator(terminal_env, ...)
   evaluator.evaluate()  # ERROR! payoff is None
   ```

## The Solution

Added a new helper function `_get_best_terminal_path()` that follows the most visited children through the MCTS tree until reaching terminal depth:

```python
def _get_best_terminal_path(mcts, target_depth):
    """Find the best complete path through the MCTS tree to a terminal node.

    Follows the most visited children from root until reaching target depth.

    Args:
        mcts: MCTS instance with completed search
        target_depth: Depth to reach (typically num_stages for the game)

    Returns:
        Tuple representing the complete path (e.g., (2, 1, 4) for 3 stages)
    """
    current_abstract = ()

    for depth in range(target_depth):
        # Get children of current node
        if current_abstract not in mcts.tree or len(mcts.tree[current_abstract]) == 0:
            raise ValueError(f"No children found at depth {depth}. Tree may not be deep enough.")

        # Select most visited child
        children = mcts.tree[current_abstract]
        best_child = max(children, key=lambda child: mcts.visits[child])
        current_abstract = best_child

    return current_abstract
```

## How It Works

### Step-by-Step Execution

Starting from root `()`:

1. **Depth 0 → 1**:
   - Children of `()`: `[(0,), (1,), (2,), (3,)]`
   - Visits: `{(0,): 10, (1,): 5, (2,): 30, (3,): 5}`
   - Select most visited: `(2,)` with 30 visits
   - `current_abstract = (2,)`

2. **Depth 1 → 2**:
   - Children of `(2,)`: `[(2, 0), (2, 1), (2, 2)]`
   - Visits: `{(2, 0): 8, (2, 1): 18, (2, 2): 4}`
   - Select most visited: `(2, 1)` with 18 visits
   - `current_abstract = (2, 1)`

3. **Depth 2 → 3**:
   - Children of `(2, 1)`: `[(2, 1, 3), (2, 1, 4), (2, 1, 5)]`
   - Visits: `{(2, 1, 3): 4, (2, 1, 4): 10, (2, 1, 5): 4}`
   - Select most visited: `(2, 1, 4)` with 10 visits
   - `current_abstract = (2, 1, 4)`

4. **Return**: `(2, 1, 4)` ✓ (depth 3 = terminal)

### Updated Runner Functions

Changed from:
```python
# Get best action sequence
best_child, _ = mcts.select_action()  # Depth 1 only!

# Reconstruct terminal environment from tree path
terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_child)
```

To:
```python
# Get best complete path to terminal node
best_terminal_path = _get_best_terminal_path(mcts, env.num_stages)  # Full depth!

# Reconstruct terminal environment from tree path
terminal_env = _reconstruct_terminal_from_tree_path(env, mcts, best_terminal_path)
```

Now:
- `best_terminal_path` has depth = `num_stages` (e.g., 3)
- `terminal_env.terminal = True`
- `terminal_env.payoff` is computed
- `Evaluator` can compute regret without errors

## Files Modified

### `tests/test_loss_game_solvers.py`

1. **Added new function** (lines 105-129):
   - `_get_best_terminal_path(mcts, target_depth)`

2. **Updated three functions**:
   - `run_mcts_progressive()` - line ~193
   - `run_gradient_mcts_progressive()` - line ~240
   - `run_mcts_local_update_progressive()` - line ~287

All three now use `_get_best_terminal_path()` instead of `select_action()`.

## Verification

### Quick Test
```bash
python3 test_terminal_path_fix.py
```

This test:
1. Runs MCTS with 50 simulations
2. Compares old way (shallow) vs new way (terminal)
3. Verifies the terminal state has a computed payoff
4. Should print "✓ SUCCESS!"

### Expected Output
```
Testing Terminal Path Fix
======================================================================

Game: LossGame1DEnv
  num_stages: 3
  num_players: 3

Running MCTS with 50 simulations...

[OLD WAY] Using select_action() - returns immediate best child:
  best_child: (2,)
  depth: 1

[NEW WAY] Using _get_best_terminal_path() - returns complete path:
  best_terminal_path: (2, 1, 4)
  depth: 3

[VERIFICATION]
  terminal_env.terminal: True
  terminal_env.payoff: tensor([...])
  len(terminal_env.state): 3

✓ SUCCESS! The fix works correctly.
  - Found complete path to terminal node
  - Terminal state has payoff: [...]
```

## Why This Fix is Correct

### 1. Follows MCTS Policy Completely
- MCTS builds a policy (visit counts) over the entire tree
- The fix follows this policy to its logical conclusion
- Most visited path = best trajectory found by MCTS

### 2. Semantically Correct for Evaluation
- We want to evaluate: "How good is the solution MCTS found?"
- The solution is a complete trajectory, not just the first move
- This fix extracts that complete trajectory

### 3. Matches Standard Practice
- Similar to how MCTS is used in game-playing (e.g., AlphaGo)
- Follow the principal variation (PV) to see the full plan
- PV = path of most visited nodes

## Potential Issues and Solutions

### Issue 1: Tree Not Deep Enough
If MCTS didn't explore deeply enough:
```
ValueError: No children found at depth 2. Tree may not be deep enough.
```

**Solutions:**
- Increase `num_simulations`
- Reduce `widening_factor` (forces deeper exploration)
- The error message is informative for debugging

### Issue 2: Multiple Equally Visited Paths
If ties in visit counts:
```python
best_child = max(children, key=lambda child: mcts.visits[child])
```

**Behavior:**
- Python's `max()` returns first maximum found
- Consistent within a run (deterministic)
- Different seeds may break ties differently

**Why it's okay:**
- All tied paths have similar quality (same visit count)
- Consistent choice for reproducibility
- If needed, can add tiebreaker (e.g., by value)

## Comparison with Alternative Approaches

### Alternative 1: Complete with Rollout
```python
# If tree is shallow, complete with random actions
while not env.terminal:
    if depth in tree:
        action = most_visited_action
    else:
        action = random_action
    env = env.move(action)
```

**Pros:** Works even with shallow trees
**Cons:** Introduces randomness; not using MCTS policy fully

**Why we didn't use it:** Our MCTS should always explore to full depth with enough simulations.

### Alternative 2: Store Terminal States in MCTS
```python
# Modify MCTS to track terminal states
mcts.terminal_states = [...]
best_terminal = max(mcts.terminal_states, key=visits)
```

**Pros:** Cleaner interface
**Cons:** Requires modifying MCTS class; memory overhead

**Why we didn't use it:** Wanted minimal changes to existing codebase.

## Physics/ML Analogy

For someone with physics background:

- **MCTS Tree**: Like a wavefunction representing superposition of possible trajectories
- **Visit Counts**: Like probability amplitude squared (frequency of visiting)
- **Best Terminal Path**: Like finding the classical path (most probable trajectory)
- **Our Fix**: Collapses the superposition to the most likely complete classical trajectory

In ML terms:
- **MCTS**: Learns a policy distribution over action sequences
- **Visit Counts**: Approximation of policy strength
- **Best Terminal Path**: Maximum likelihood trajectory under learned policy

## Summary

The fix ensures that we properly extract complete terminal trajectories from the MCTS tree for evaluation, rather than just getting the first action. This allows the `Evaluator` to compute regret correctly by comparing complete solutions.

**Key changes:**
1. Added `_get_best_terminal_path()` to find complete paths
2. Updated all MCTS runner functions to use it
3. Now returns terminal states with computed payoffs
4. Evaluator works without errors
