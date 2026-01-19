"""
Analysis and visualization tools for loss game solver comparison results.

This module provides functions to load experimental results and generate
comparison visualizations and summary statistics.
"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Data Loading
# =============================================================================

def load_results(filename):
    """Load pickled results from test run.

    Args:
        filename: Path to pickle file

    Returns:
        Dictionary containing results and metadata
    """
    with open(filename, "rb") as f:
        payload = pickle.load(f)
    return payload


def extract_metrics(results_dict):
    """Extract metrics from results for plotting.

    Args:
        results_dict: Dictionary mapping solver names to result lists

    Returns:
        Dictionary mapping solver names to extracted metric dictionaries
    """
    metrics = {}
    for solver_name, solver_results in results_dict.items():
        iterations = [r["iteration"] for r in solver_results]
        payoffs = [r["payoff"] for r in solver_results]  # Shape: (n_checkpoints, n_players)
        regrets = [r["regret"] for r in solver_results]  # Shape: (n_checkpoints, n_players)
        times = [r["time"] for r in solver_results]

        metrics[solver_name] = {
            "iterations": iterations,
            "payoffs": payoffs,
            "regrets": regrets,
            "times": times,
        }
    return metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_payoff_curves(metrics, game_name, player_idx=0):
    """Plot payoff vs iterations/simulations for each solver.

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game
        player_idx: Which player to plot (default: 0)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'MCTS': 'blue', 'GradientMCTS': 'green',
              'MCTSwithLocalUpdate': 'orange', 'LocalStochasticMinimaxUpdate': 'red'}

    for solver_name, data in metrics.items():
        iterations = data["iterations"]
        payoffs = [p[player_idx].item() for p in data["payoffs"]]
        color = colors.get(solver_name, 'black')
        ax.plot(iterations, payoffs, marker='o', label=solver_name,
                linewidth=2, color=color, markersize=6)

    ax.set_xlabel("Iterations / Simulations", fontsize=12)
    ax.set_ylabel(f"Payoff (Player {player_idx})", fontsize=12)
    ax.set_title(f"{game_name}: Payoff Convergence", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_regret_curves(metrics, game_name, player_idx=0):
    """Plot regret vs iterations/simulations for each solver.

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game
        player_idx: Which player to plot (default: 0)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'MCTS': 'blue', 'GradientMCTS': 'green',
              'MCTSwithLocalUpdate': 'orange', 'LocalStochasticMinimaxUpdate': 'red'}

    for solver_name, data in metrics.items():
        iterations = data["iterations"]
        regrets = [r[player_idx].item() for r in data["regrets"]]
        color = colors.get(solver_name, 'black')
        ax.plot(iterations, regrets, marker='o', label=solver_name,
                linewidth=2, color=color, markersize=6)

    ax.set_xlabel("Iterations / Simulations", fontsize=12)
    ax.set_ylabel(f"Regret (Player {player_idx})", fontsize=12)
    ax.set_title(f"{game_name}: Regret Convergence (lower is better)", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_final_comparison(metrics, game_name):
    """Bar chart comparing final metrics across solvers.

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    solvers = list(metrics.keys())
    final_payoffs_p0 = [data["payoffs"][-1][0].item() for data in metrics.values()]
    final_regrets_p0 = [data["regrets"][-1][0].item() for data in metrics.values()]

    colors = ['blue', 'green', 'orange', 'red'][:len(solvers)]

    # Payoff comparison
    bars1 = ax1.bar(range(len(solvers)), final_payoffs_p0, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(range(len(solvers)))
    ax1.set_xticklabels(solvers, rotation=45, ha='right')
    ax1.set_ylabel("Final Payoff (Player 0)", fontsize=12)
    ax1.set_title("Final Payoff Comparison", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, final_payoffs_p0)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Regret comparison
    bars2 = ax2.bar(range(len(solvers)), final_regrets_p0, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(solvers)))
    ax2.set_xticklabels(solvers, rotation=45, ha='right')
    ax2.set_ylabel("Final Regret (Player 0)", fontsize=12)
    ax2.set_title("Final Regret Comparison (lower is better)", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, final_regrets_p0)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f"{game_name}: Final Metrics", fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_efficiency(metrics, game_name, player_idx=0):
    """Plot payoff improvement vs computational cost (time).

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game
        player_idx: Which player to plot (default: 0)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'MCTS': 'blue', 'GradientMCTS': 'green',
              'MCTSwithLocalUpdate': 'orange', 'LocalStochasticMinimaxUpdate': 'red'}

    for solver_name, data in metrics.items():
        times = data["times"]
        payoffs = [p[player_idx].item() for p in data["payoffs"]]
        color = colors.get(solver_name, 'black')
        ax.plot(times, payoffs, marker='o', label=solver_name,
                linewidth=2, color=color, markersize=6)

    ax.set_xlabel("Wall Clock Time (seconds)", fontsize=12)
    ax.set_ylabel(f"Payoff (Player {player_idx})", fontsize=12)
    ax.set_title(f"{game_name}: Computational Efficiency", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_all_players(metrics, game_name, num_players=3):
    """Plot regret curves for all players.

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game
        num_players: Number of players in the game (default: 3)

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, num_players, figsize=(6*num_players, 5))
    if num_players == 1:
        axes = [axes]

    colors = {'MCTS': 'blue', 'GradientMCTS': 'green',
              'MCTSwithLocalUpdate': 'orange', 'LocalStochasticMinimaxUpdate': 'red'}

    for player_idx in range(num_players):
        ax = axes[player_idx]

        for solver_name, data in metrics.items():
            iterations = data["iterations"]
            regrets = [r[player_idx].item() for r in data["regrets"]]
            color = colors.get(solver_name, 'black')
            ax.plot(iterations, regrets, marker='o', label=solver_name,
                    linewidth=2, color=color, markersize=4)

        ax.set_xlabel("Iterations / Simulations", fontsize=11)
        ax.set_ylabel(f"Regret", fontsize=11)
        ax.set_title(f"Player {player_idx}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{game_name}: Regret Convergence (All Players)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_all_plots(filename, output_dir="plots"):
    """Load results and generate all plots.

    Args:
        filename: Path to results pickle file
        output_dir: Directory to save plots (default: "plots")

    Returns:
        None (saves plots to disk and displays them)
    """
    payload = load_results(filename)
    game_name = payload["game_name"]
    results = payload["results"]

    metrics = extract_metrics(results)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating plots for {game_name}...")

    # Generate and save all plots
    fig1 = plot_payoff_curves(metrics, game_name, player_idx=0)
    fig1.savefig(f"{output_dir}/{game_name}_payoff_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{game_name}_payoff_curves.png")

    fig2 = plot_regret_curves(metrics, game_name, player_idx=0)
    fig2.savefig(f"{output_dir}/{game_name}_regret_curves.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{game_name}_regret_curves.png")

    fig3 = plot_final_comparison(metrics, game_name)
    fig3.savefig(f"{output_dir}/{game_name}_final_comparison.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{game_name}_final_comparison.png")

    fig4 = plot_efficiency(metrics, game_name, player_idx=0)
    fig4.savefig(f"{output_dir}/{game_name}_efficiency.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{game_name}_efficiency.png")

    fig5 = plot_all_players(metrics, game_name, num_players=3)
    fig5.savefig(f"{output_dir}/{game_name}_all_players.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir}/{game_name}_all_players.png")

    print(f"\nAll plots saved to {output_dir}/")
    plt.show()


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_statistics(metrics, game_name):
    """Print summary statistics for all solvers.

    Args:
        metrics: Dictionary of extracted metrics
        game_name: Name of the game

    Returns:
        None (prints to console)
    """
    print("=" * 80)
    print(f"Summary Statistics: {game_name}")
    print("=" * 80)

    for solver_name, data in metrics.items():
        print(f"\n{solver_name}:")
        print(f"  Final iteration: {data['iterations'][-1]}")

        # Print stats for all players
        num_players = len(data['payoffs'][-1])
        for player_idx in range(num_players):
            final_payoff = data['payoffs'][-1][player_idx].item()
            final_regret = data['regrets'][-1][player_idx].item()
            print(f"  Player {player_idx}:")
            print(f"    Final payoff: {final_payoff:.4f}")
            print(f"    Final regret: {final_regret:.4f}")

        print(f"  Total time: {data['times'][-1]:.2f}s")

        # Convergence analysis (for player 0)
        regrets_p0 = [r[0].item() for r in data["regrets"]]
        if len(regrets_p0) >= 2:
            improvement = regrets_p0[0] - regrets_p0[-1]
            print(f"  Regret improvement (P0): {improvement:.4f}")

            # Calculate convergence rate
            if improvement > 0:
                iterations_range = data['iterations'][-1] - data['iterations'][0]
                if iterations_range > 0:
                    rate = improvement / iterations_range
                    print(f"  Convergence rate (P0): {rate:.6f} per iteration")

    print("=" * 80)


def compare_solvers_summary(metrics):
    """Create a comparative summary table of all solvers.

    Args:
        metrics: Dictionary of extracted metrics

    Returns:
        None (prints to console)
    """
    print("\n" + "=" * 80)
    print("Comparative Summary (Player 0 metrics)")
    print("=" * 80)
    print(f"{'Solver':<30} {'Final Payoff':<15} {'Final Regret':<15} {'Time (s)':<15}")
    print("-" * 80)

    for solver_name, data in metrics.items():
        final_payoff = data['payoffs'][-1][0].item()
        final_regret = data['regrets'][-1][0].item()
        total_time = data['times'][-1]

        print(f"{solver_name:<30} {final_payoff:<15.4f} {final_regret:<15.4f} {total_time:<15.2f}")

    print("=" * 80 + "\n")


# =============================================================================
# Main Entry Point for Direct Execution
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze loss game solver results")
    parser.add_argument("results_file", type=str, help="Path to results pickle file")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory for plots")
    parser.add_argument("--no-display", action="store_true", help="Don't display plots (only save)")

    args = parser.parse_args()

    # Load and extract metrics
    print(f"Loading results from {args.results_file}...")
    payload = load_results(args.results_file)
    metrics = extract_metrics(payload["results"])

    # Print summary statistics
    print_summary_statistics(metrics, payload["game_name"])
    compare_solvers_summary(metrics)

    # Generate plots
    if args.no_display:
        # Prevent display but still save
        import matplotlib
        matplotlib.use('Agg')

    generate_all_plots(args.results_file, args.output_dir)
