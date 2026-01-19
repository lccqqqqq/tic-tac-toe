"""
High-level runner script for loss game solver comparison experiments.

This script provides a convenient interface to run comprehensive experiments
comparing different solver algorithms on loss game environments.
"""

import argparse
import sys
import os

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

from test_loss_game_solvers import compare_solvers_on_game, save_results
from analyze_loss_game_results import (generate_all_plots, print_summary_statistics,
                                       compare_solvers_summary, extract_metrics, load_results)


def main():
    parser = argparse.ArgumentParser(
        description="Run loss game solver comparison experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all solvers on both games and analyze
  python run_loss_game_experiments.py --game both --analyze

  # Run only MCTS and GradientMCTS on 1D game
  python run_loss_game_experiments.py --game LossGame1D --solvers MCTS GradientMCTS

  # Run with custom seed
  python run_loss_game_experiments.py --game LossGame1D --seed 123

  # Analyze existing results only
  python run_loss_game_experiments.py --analyze-only Data/loss_game_comparison/LossGame1D_20260119_143022.pkl
        """
    )

    parser.add_argument("--game", choices=["LossGame1D", "LossGame3D", "both"], default="both",
                       help="Which game variant to test (default: both)")
    parser.add_argument("--solvers", nargs="+",
                       choices=["MCTS", "GradientMCTS", "MCTSwithLocalUpdate",
                               "LocalStochasticMinimaxUpdate", "all"],
                       default=["all"],
                       help="Which solvers to run (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--analyze", action="store_true",
                       help="Run analysis and generate plots after experiments")
    parser.add_argument("--analyze-only", type=str, metavar="RESULTS_FILE",
                       help="Skip experiments and only analyze existing results file")
    parser.add_argument("--output-dir", type=str, default="plots",
                       help="Output directory for plots (default: plots)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display plots (only save to disk)")

    args = parser.parse_args()

    # Handle analyze-only mode
    if args.analyze_only:
        print(f"\n{'='*80}")
        print(f"ANALYZE-ONLY MODE")
        print(f"{'='*80}\n")

        if not os.path.exists(args.analyze_only):
            print(f"ERROR: Results file not found: {args.analyze_only}")
            sys.exit(1)

        print(f"Loading results from {args.analyze_only}...")
        payload = load_results(args.analyze_only)
        metrics = extract_metrics(payload["results"])

        # Print statistics
        print_summary_statistics(metrics, payload["game_name"])
        compare_solvers_summary(metrics)

        # Generate plots
        if args.no_display:
            import matplotlib
            matplotlib.use('Agg')

        output_dir = os.path.join(args.output_dir, payload["game_name"])
        generate_all_plots(args.analyze_only, output_dir=output_dir)

        print(f"\n{'='*80}")
        print(f"Analysis complete!")
        print(f"{'='*80}\n")
        return

    # Determine which solvers to run
    if "all" in args.solvers:
        solvers_to_run = ["MCTS", "GradientMCTS", "MCTSwithLocalUpdate", "LocalStochasticMinimaxUpdate"]
    else:
        solvers_to_run = args.solvers

    # Determine which games to run
    if args.game == "both":
        games = ["LossGame1D", "LossGame3D"]
    else:
        games = [args.game]

    print(f"\n{'='*80}")
    print(f"LOSS GAME SOLVER COMPARISON")
    print(f"{'='*80}")
    print(f"Games to test: {', '.join(games)}")
    print(f"Solvers to run: {', '.join(solvers_to_run)}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")

    # Run experiments
    result_files = []
    for game_name in games:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS ON {game_name}")
        print(f"{'='*80}\n")

        results = compare_solvers_on_game(game_name, solvers_to_run, seed=args.seed)
        filename = save_results(results, game_name)
        result_files.append(filename)

        # Optionally analyze immediately
        if args.analyze:
            print(f"\n{'='*80}")
            print(f"ANALYZING RESULTS FOR {game_name}")
            print(f"{'='*80}\n")

            metrics = extract_metrics(results)
            print_summary_statistics(metrics, game_name)
            compare_solvers_summary(metrics)

            # Generate plots
            if args.no_display:
                import matplotlib
                matplotlib.use('Agg')

            output_dir = os.path.join(args.output_dir, game_name)
            generate_all_plots(filename, output_dir=output_dir)

    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResult files:")
    for rf in result_files:
        print(f"  - {rf}")

    if not args.analyze:
        print(f"\nTo analyze results later, run:")
        for rf in result_files:
            print(f"  python run_loss_game_experiments.py --analyze-only {rf}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
