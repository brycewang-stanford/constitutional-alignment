#!/usr/bin/env python3
"""
ValuePriorityBench: Main Experiment Runner
Runs the complete experiment pipeline for value priority inference.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from collect_responses import run_experiment, MODELS, ModelConfig
from parse_responses import parse_all_responses, generate_comparison_data
from bayesian_inference import run_inference, compute_priority_alignment_score, compute_weighted_pas
from visualize import generate_all_figures

def main():
    parser = argparse.ArgumentParser(description="Run ValuePriorityBench experiment")
    parser.add_argument("--collect", action="store_true", help="Collect responses from LLM APIs")
    parser.add_argument("--parse", action="store_true", help="Parse collected responses")
    parser.add_argument("--infer", action="store_true", help="Run Bayesian inference")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--responses-file", type=str, help="Specific responses file to use")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per scenario")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    scenarios_path = base_dir / "data" / "scenarios" / "scenarios.json"
    responses_dir = base_dir / "data" / "responses"
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures"

    # Create directories
    responses_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        args.collect = args.parse = args.infer = args.visualize = True

    # Step 1: Collect responses
    if args.collect:
        print("\n" + "="*70)
        print("STEP 1: Collecting Responses from LLM APIs")
        print("="*70)

        responses = run_experiment(
            str(scenarios_path),
            str(responses_dir),
            models=MODELS,
            repetitions=args.repetitions,
            delay_between_calls=args.delay
        )

    # Step 2: Parse responses
    if args.parse:
        print("\n" + "="*70)
        print("STEP 2: Parsing Responses")
        print("="*70)

        # Find latest responses file
        if args.responses_file:
            responses_path = Path(args.responses_file)
        else:
            response_files = list(responses_dir.glob("responses_*.json"))
            if not response_files:
                print("ERROR: No response files found!")
                return
            responses_path = max(response_files, key=lambda p: p.stat().st_mtime)

        print(f"Using responses file: {responses_path}")

        parsed_path = responses_dir / "parsed_choices.json"
        parsed = parse_all_responses(
            str(responses_path),
            str(scenarios_path),
            str(parsed_path)
        )

        # Generate comparison data
        comparisons = generate_comparison_data(parsed, str(scenarios_path))
        comparisons_path = responses_dir / "comparisons.json"
        with open(comparisons_path, 'w') as f:
            json.dump(comparisons, f, indent=2)
        print(f"Generated {len(comparisons)} pairwise comparisons")

    # Step 3: Run Bayesian inference
    if args.infer:
        print("\n" + "="*70)
        print("STEP 3: Running Bayesian Bradley-Terry Inference")
        print("="*70)

        comparisons_path = responses_dir / "comparisons.json"
        if not comparisons_path.exists():
            print("ERROR: comparisons.json not found. Run --parse first.")
            return

        inference_path = results_dir / "inference_results.json"
        results = run_inference(
            str(comparisons_path),
            str(inference_path),
            use_pymc=False  # Use analytical method for speed
        )

        # Generate summary report
        generate_summary_report(results, results_dir / "summary_report.md")

    # Step 4: Generate visualizations
    if args.visualize:
        print("\n" + "="*70)
        print("STEP 4: Generating Visualizations")
        print("="*70)

        inference_path = results_dir / "inference_results.json"
        parsed_path = responses_dir / "parsed_choices.json"

        if not inference_path.exists():
            print("ERROR: inference_results.json not found. Run --infer first.")
            return

        generate_all_figures(
            str(inference_path),
            str(parsed_path),
            str(figures_dir)
        )

    print("\n" + "="*70)
    print("Experiment Complete!")
    print("="*70)

def get_result_attr(result, attr):
    """Helper to get attribute from either dict or object."""
    if hasattr(result, attr):
        return getattr(result, attr)
    return result[attr]

def generate_summary_report(results: dict, output_path: Path):
    """Generate a markdown summary report of the results."""
    claude_declared = ["safety", "honesty", "compliance", "helpfulness"]

    report = []
    report.append("# ValuePriorityBench Results Summary")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("## Priority Orderings by Model\n")
    report.append("| Model | Inferred Priority Ordering |")
    report.append("|-------|---------------------------|")

    for model_id, result in results.items():
        ordering = " > ".join(get_result_attr(result, 'priority_ordering'))
        report.append(f"| {model_id.title()} | {ordering} |")

    report.append("\n## Priority Alignment Scores (vs Claude Constitution)\n")
    report.append("Claude's declared ordering: Safety > Honesty > Compliance > Helpfulness\n")
    report.append("| Model | PAS | Weighted PAS | Match with Claude |")
    report.append("|-------|-----|--------------|-------------------|")

    for model_id, result in results.items():
        priority_ordering = get_result_attr(result, 'priority_ordering')
        pas, tau = compute_priority_alignment_score(
            priority_ordering,
            claude_declared
        )
        weighted_pas = compute_weighted_pas(
            priority_ordering,
            claude_declared
        )
        match = "Yes" if priority_ordering == claude_declared else "No"
        report.append(f"| {model_id.title()} | {pas:.3f} | {weighted_pas:.3f} | {match} |")

    report.append("\n## Pairwise Dominance Probabilities\n")

    for model_id, result in results.items():
        report.append(f"\n### {model_id.title()}\n")
        report.append("| Value i | Value j | P(i > j) |")
        report.append("|---------|---------|----------|")

        pairwise = get_result_attr(result, 'pairwise_probabilities')
        for p in pairwise:
            if hasattr(p, 'value_i'):
                vi, vj, prob = p.value_i, p.value_j, p.probability
            else:
                vi, vj, prob = p['value_i'], p['value_j'], p['probability']
            report.append(f"| {vi.title()} | {vj.title()} | {prob:.3f} |")

    report.append("\n## Convergence Diagnostics\n")
    report.append("| Model | Method | N Comparisons |")
    report.append("|-------|--------|---------------|")

    for model_id, result in results.items():
        diag = get_result_attr(result, 'convergence_diagnostics')
        report.append(f"| {model_id.title()} | {diag['method']} | {diag['n_comparisons']} |")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Summary report saved to: {output_path}")

if __name__ == "__main__":
    main()
