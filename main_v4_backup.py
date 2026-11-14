"""
Main Pipeline Orchestrator
=========================
Runs the complete AI-powered cloud resource optimization pipeline.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_simulation import generate_synthetic_cloud_data
from dataset_adapter import load_google_cluster_dataset
from preprocessing import DataPreprocessor
from model_training import LSTMModel, RandomForestModel
from prediction import WorkloadPredictor
from optimization_engine import ResourceOptimizer, format_indian_currency
from autoscaler import AutoScaler
from cloud_simulator import CloudSimulator
from visualization import Visualizer
from decision_logger import DecisionLogger

import pandas as pd
import numpy as np


def main():
    """
    Main function to run the complete pipeline.
    """
    print("\n" + "="*70)
    print("AI-POWERED CLOUD RESOURCE OPTIMIZATION FRAMEWORK — V4")
    print("="*70 + "\n")
    
    # Configuration
    DATA_FILE = "data/google_cluster_usage.csv"
    FEATURE_COLUMNS = ["cpu_usage", "ram_usage", "assigned_mem_usage"]
    TARGET_COLUMNS = ["cpu_usage", "ram_usage"]
    MODEL_TYPE = "lstm"  # Options: 'lstm' or 'rf'
    SEQUENCE_LENGTH = 24
    EPOCHS = 30  # Reduced for faster execution
    BATCH_SIZE = 32
    
    # Step 1: Load the public Google Cluster aggregate workload dataset
    print("📊 STEP 1: Dataset Loading")
    print("-"*70)
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Download it before running the pipeline."
        )
    df = load_google_cluster_dataset(DATA_FILE)
    print()

    # Step 2: Preprocess data
    print("🔧 STEP 2: Data Preprocessing")
    print("-"*70)
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    
    if MODEL_TYPE == 'lstm':
        X_train, X_test, y_train, y_test = preprocessor.prepare_lstm_data(
            df, sequence_length=SEQUENCE_LENGTH,
            features=FEATURE_COLUMNS, targets=TARGET_COLUMNS
        )
    else:
        X_train, X_test, y_train, y_test = preprocessor.prepare_rf_data(
            df, lookback=SEQUENCE_LENGTH
        )
    
    # Split train into train and validation
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    print()
    
    # Step 3: Train model
    print("🤖 STEP 3: Model Training")
    print("-"*70)
    model_path = (
        f"models/google_cluster_{MODEL_TYPE}_model.h5"
        if MODEL_TYPE == 'lstm' else f"models/google_cluster_{MODEL_TYPE}_model.pkl"
    )
    
    if MODEL_TYPE == 'lstm':
        if os.path.exists(model_path):
            try:
                print(f"✅ Loading existing LSTM model from {model_path}")
                model = LSTMModel(sequence_length=SEQUENCE_LENGTH, n_features=3, n_targets=2)
                model.load(model_path)
            except (ValueError, Exception) as e:
                print(f"⚠️  Failed to load model: {e}")
                print("🔄 Retraining LSTM model...")
                # Delete the incompatible model file
                if os.path.exists(model_path):
                    os.remove(model_path)
                model = LSTMModel(sequence_length=SEQUENCE_LENGTH, n_features=3, n_targets=2)
                model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
                model.train(X_train, y_train, X_val, y_val, 
                           epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
                model.save(model_path)
        else:
            print("🔄 Training new LSTM model...")
            model = LSTMModel(sequence_length=SEQUENCE_LENGTH, n_features=3, n_targets=2)
            model.build_model(lstm_units=[64, 32], dropout_rate=0.2)
            model.train(X_train, y_train, X_val, y_val, 
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
            model.save(model_path)
        
        # Evaluate model
        print("\n📊 Model Evaluation:")
        model.evaluate(X_test, y_test, preprocessor.target_scaler)
    else:
        if os.path.exists(model_path):
            print(f"✅ Loading existing Random Forest model from {model_path}")
            model = RandomForestModel()
            model.load(model_path)
        else:
            print("🔄 Training new Random Forest model...")
            model = RandomForestModel(n_estimators=100, max_depth=20)
            model.train(X_train, y_train)
            model.save(model_path)
        
        # Evaluate model
        print("\n📊 Model Evaluation:")
        model.evaluate(X_test, y_test, preprocessor.target_scaler)
    print()
    
    # Step 4: Initialize components
    print("⚙️  STEP 4: Initializing Components")
    print("-"*70)
    optimizer = ResourceOptimizer()
    autoscaler = AutoScaler()
    predictor = WorkloadPredictor(model_type=MODEL_TYPE, model_path=model_path)
    simulator = CloudSimulator(optimizer=optimizer, autoscaler=autoscaler)
    visualizer = Visualizer(output_dir='outputs')
    print("✅ All components initialized")
    print()
    
    # Step 5: Run simulations with Scenarios
    print("🔄 STEP 5: Running Scenario-Based Simulations")
    print("-"*70)
    
    # Import Scenarios
    from scenarios import WorkloadScenario
    
    # Define scenarios to test
    scenarios_list = [
        "Baseline (Normal)",
        "Sudden Traffic Spike",
        "Night-Time Low Load",
        "Flash Sale Scenario",
        "Gradual Growth"
    ]
    
    # Fixed held-out workload window. AI uses the first 24 rows as warm-up;
    # every strategy is scored on the same remaining rows.
    np.random.seed(42)
    base_test_data = df.tail(500).copy()
    
    scenario_results_summary = []
    
    # Initialize decision logger for AI explainability
    decision_logger = DecisionLogger(output_file="outputs/decision_timeline.txt")

    print(f"Testing {len(scenarios_list)} scenarios...\n")

    for scenario_name in scenarios_list:
        print(f"🎬 SCENARIO: {scenario_name}")
        print("="*50)
        
        # Apply scenario
        if scenario_name == "Baseline (Normal)":
            test_data = base_test_data.copy()
        else:
            test_data = WorkloadScenario.apply_scenario(base_test_data, scenario_name)
        
        evaluation_data = test_data.iloc[SEQUENCE_LENGTH:].copy()
        if evaluation_data.empty:
            raise ValueError("Scenario data is too short after AI warm-up.")

        print("   Running Fixed Allocation Strategy...")
        baseline_results = simulator.simulate_baseline(
            evaluation_data, fixed_cpu=100, fixed_ram=100
        )

        print("   Running Threshold-Based Scaler...")
        threshold_results = simulator.simulate_threshold_scaling(evaluation_data)

        print("   Running AI-Optimized Strategy...")
        ai_results = simulator.simulate_ai_optimized(
            test_data, predictor, preprocessor,
            lookback_window=SEQUENCE_LENGTH,
            decision_logger=decision_logger
        )

        comparison = simulator.compare_strategies(baseline_results, ai_results)
        
        # Store for summary
        summary_entry = {
            "Scenario": scenario_name,
            "Baseline Cost": baseline_results['total_cost'],
            "Threshold Cost": threshold_results['total_cost'],
            "AI Cost": ai_results['total_cost'],
            "Cost Saving (%)": comparison['cost_reduction'],
            "AI Efficiency": ai_results['avg_efficiency'],
            "AI Avg Instances": ai_results.get('avg_instance_count', 0.0),
            "AI Max Instances": ai_results.get('max_instance_count', 0),
            "AI Prediction Failures": ai_results.get('prediction_failures', 0),
            "AI Scale Ups": ai_results.get('scale_up_events', 0),
            "AI Scale Downs": ai_results.get('scale_down_events', 0),
            "Imp. (%)": comparison['efficiency_improvement'],
            "Baseline CPU": baseline_results['avg_cpu_utilization'],
            "Threshold CPU": threshold_results['avg_cpu_utilization'],
            "AI CPU": ai_results['avg_cpu_utilization'],
            "Baseline RAM": baseline_results['avg_ram_utilization'],
            "Threshold RAM": threshold_results['avg_ram_utilization'],
            "AI RAM": ai_results['avg_ram_utilization'],
            "Baseline SLA": baseline_results['sla_violation_rate'],
            "Threshold SLA": threshold_results['sla_violation_rate'],
            "AI SLA": ai_results['sla_violation_rate'],
            "Baseline Efficiency": baseline_results['avg_efficiency'],
            "Threshold Efficiency": threshold_results['avg_efficiency']
        }
        scenario_results_summary.append(summary_entry)
        print(f"   ✅ Scenario '{scenario_name}' completed.")
        print("-" * 50 + "\n")

    
    # Compare strategies (using the last run for default report generation or aggregate? 
    # The prompt asks for scenario-wise results to outputs/scenario_results.txt.
    # We will generate a specific report for scenarios.)

    print("📄 STEP 6: Generating Scenario Report")
    print("-"*70)
    
    # Write summary to log file and print to console
    decision_logger.write_summary_to_file()
    
    # Create Scenario Result Table
    print("\n" + "="*115)
    print("                           SCENARIO-BASED EVALUATION REPORT (3-WAY COMPARISON)")
    print("="*115)
    
    # Header
    header = f"{'Scenario Name':<25} | {'Base Cost':<12} | {'Thr. Cost':<12} | {'AI Cost':<12} | {'Saving (%)':<12} | {'AI Eff.':<10} | {'Imp. (%)':<12}"
    sep = "-"*120
    print(header)
    print(sep)
    
    report_lines = [
        "="*115,
        "                           SCENARIO-BASED EVALUATION REPORT (3-WAY COMPARISON)",
        "="*115,
        header,
        sep
    ]
    
    for res in scenario_results_summary:
        base_cost_str = format_indian_currency(res['Baseline Cost'])
        thr_cost_str = format_indian_currency(res['Threshold Cost'])
        ai_cost_str = format_indian_currency(res['AI Cost'])
        line = f"{res['Scenario']:<25} | {base_cost_str:<12} | {thr_cost_str:<12} | {ai_cost_str:<12} | {res['Cost Saving (%)']:<12.2f} | {res['AI Efficiency']:<10.4f} | {res['Imp. (%)']:<12.2f}"
        print(line)
        report_lines.append(line)
        diagnostics = (f"    AI fleet: avg={res.get('AI Avg Instances', 0):.2f}, "
                       f"max={res.get('AI Max Instances', 0)}, "
                       f"scale-ups={res.get('AI Scale Ups', 0)}, "
                       f"scale-downs={res.get('AI Scale Downs', 0)}, "
                       f"prediction-fallbacks={res.get('AI Prediction Failures', 0)}")
        report_lines.append(diagnostics)
        
    print(sep)
    report_lines.append(sep)
    
    # Save scenario results
    scenario_path = "outputs/scenario_results.txt"
    with open(scenario_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
        
    print(f"\n✅ Scenario results saved to {scenario_path}")

    # Generate plots for the last scenario (usually Baseline/Normal or Gradual Growth)
    print("📊 Generating comparison plots...")
    visualizer.plot_utilization_comparison(baseline_results, ai_results, threshold_results,
                                         save_path='outputs/utilization_comparison.png')
    visualizer.plot_cost_efficiency(baseline_results, ai_results, threshold_results,
                                  save_path='outputs/cost_efficiency.png')
    visualizer.plot_time_series_utilization(baseline_results, ai_results, threshold_results,
                                          save_path='outputs/time_series_utilization.png')

    
    # Aggregate summary across all five scenarios. Utilization/SLA/efficiency
    # are equal-weight scenario means; cost is summed across scenario runs.
    print("="*105)
    print("                 AGGREGATE EXPERIMENT SUMMARY (ALL SCENARIOS)")
    print("="*105)

    def mean_metric(key):
        return float(np.mean([entry[key] for entry in scenario_results_summary]))

    cpu_base, cpu_thr, cpu_ai = (mean_metric(k) * 100 for k in ("Baseline CPU", "Threshold CPU", "AI CPU"))
    ram_base, ram_thr, ram_ai = (mean_metric(k) * 100 for k in ("Baseline RAM", "Threshold RAM", "AI RAM"))
    sla_base, sla_thr, sla_ai = (mean_metric(k) for k in ("Baseline SLA", "Threshold SLA", "AI SLA"))
    cost_base = sum(x["Baseline Cost"] for x in scenario_results_summary)
    cost_thr = sum(x["Threshold Cost"] for x in scenario_results_summary)
    cost_ai = sum(x["AI Cost"] for x in scenario_results_summary)
    eff_base, eff_thr, eff_ai = (mean_metric(k) for k in ("Baseline Efficiency", "Threshold Efficiency", "AI Efficiency"))

    def fmt_pct(val): return f"{val:.2f}%"
    def fmt_num(val): return f"{val:.4f}"
    def fmt_cost(val): return format_indian_currency(val)
    def get_best(base, thr, ai, metric_type="high"):
        vals = {"Baseline": base, "Threshold": thr, "AI": ai}
        return (max if metric_type == "high" else min)(vals, key=vals.get)

    header = f"{'Metric':<25} | {'Baseline':<12} | {'Threshold':<12} | {'AI-Based':<12} | {'Best Strategy':<15}"
    sep = "-" * 105
    rows = [
        f"{'Avg CPU Utilization':<25} | {fmt_pct(cpu_base):<12} | {fmt_pct(cpu_thr):<12} | {fmt_pct(cpu_ai):<12} | {get_best(cpu_base,cpu_thr,cpu_ai):<15}",
        f"{'Avg RAM Utilization':<25} | {fmt_pct(ram_base):<12} | {fmt_pct(ram_thr):<12} | {fmt_pct(ram_ai):<12} | {get_best(ram_base,ram_thr,ram_ai):<15}",
        f"{'SLA Violation Rate':<25} | {fmt_pct(sla_base):<12} | {fmt_pct(sla_thr):<12} | {fmt_pct(sla_ai):<12} | {get_best(sla_base,sla_thr,sla_ai,'low'):<15}",
        f"{'Total Cost (INR)':<25} | {fmt_cost(cost_base):<12} | {fmt_cost(cost_thr):<12} | {fmt_cost(cost_ai):<12} | {get_best(cost_base,cost_thr,cost_ai,'low'):<15}",
        f"{'Efficiency Score':<25} | {fmt_num(eff_base):<12} | {fmt_num(eff_thr):<12} | {fmt_num(eff_ai):<12} | {get_best(eff_base,eff_thr,eff_ai):<15}"
    ]
    table_lines = [
        "="*105,
        "                 AGGREGATE EXPERIMENT SUMMARY (ALL SCENARIOS)",
        "="*105, header, sep, *rows, sep,
        "Note: utilization, SLA and efficiency are equal-weight means across scenarios.",
        "Note: total cost is summed across five independent scenario experiments; it is not a single deployment bill.",
        "V4 model: CPU/RAM allocations are elastic pooled quantities; instance count is sized separately.",
        "Instance count affects base instance cost; it does not multiply elastic CPU/RAM quantities."
    ]
    summary_text = "\n".join(table_lines)
    print(summary_text)
    summary_path = "outputs/final_experiment_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nAggregate summary saved to {summary_path}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
