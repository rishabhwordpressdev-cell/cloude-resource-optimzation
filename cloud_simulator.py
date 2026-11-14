"""
Cloud Simulation Layer
======================
Simulates cloud environment and compares resource utilization before/after AI optimization.
"""

import numpy as np
import pandas as pd
from optimization_engine import ResourceOptimizer, format_indian_currency
from autoscaler import AutoScaler
from prediction import WorkloadPredictor
from preprocessing import DataPreprocessor
from decision_logger import DecisionLogger


class CloudSimulator:
    """
    Simulates cloud infrastructure and evaluates optimization strategies.
    """
    
    def __init__(self, optimizer=None, autoscaler=None):
        """
        Initialize cloud simulator.
        
        Parameters:
        -----------
        optimizer : ResourceOptimizer
            Resource optimization engine
        autoscaler : AutoScaler
            Auto-scaling decision engine
        """
        self.optimizer = optimizer or ResourceOptimizer()
        self.autoscaler = autoscaler or AutoScaler()
        self.results = []
    
    def simulate_baseline(self, data, fixed_cpu=100, fixed_ram=100, sla_threshold=0.9):
        """
        Simulate baseline scenario (fixed allocation, no AI optimization).
         SLA Violation = CPU or RAM utilization > SLA Threshold
        """
        results = {
            'cpu_utilizations': [],
            'ram_utilizations': [],
            'wasted_cpu': [],
            'wasted_ram': [],
            'costs': [],
            'efficiency_scores': [],
            'sla_violations': 0
        }
        
        for _, row in data.iterrows():
            cpu_usage = row['cpu_usage']
            ram_usage = row['ram_usage']
            
            # Calculate utilization
            cpu_util = cpu_usage / fixed_cpu if fixed_cpu > 0 else 0
            ram_util = ram_usage / fixed_ram if fixed_ram > 0 else 0
            
            cpu_util = min(1.0, max(0.0, cpu_util))
            ram_util = min(1.0, max(0.0, ram_util))
            
            # Track SLA violation
            if cpu_util > sla_threshold or ram_util > sla_threshold:
                results['sla_violations'] += 1
            
            # Calculate wasted resources
            wasted_cpu = max(0, fixed_cpu - cpu_usage)
            wasted_ram = max(0, fixed_ram - ram_usage)
            
            # Calculate cost
            # Baseline assumes 1 big fixed instance or multiple fixed instances summing to fixed_cpu
            instance_count = 1  # One fixed-capacity baseline instance in this comparison
            
            cost = self.optimizer.calculate_cost(
                allocated_cpu=fixed_cpu,
                allocated_ram=fixed_ram,
                used_cpu=cpu_usage,
                used_ram=ram_usage,
                instance_count=instance_count
            )
            
            # Efficiency score
            avg_util = (cpu_util + ram_util) / 2
            cost_eff = 1.0 / (1.0 + cost / 100.0)
            efficiency = (avg_util * 0.7 + cost_eff * 0.3)
            
            results['cpu_utilizations'].append(cpu_util)
            results['ram_utilizations'].append(ram_util)
            results['wasted_cpu'].append(wasted_cpu)
            results['wasted_ram'].append(wasted_ram)
            results['costs'].append(cost)
            results['efficiency_scores'].append(efficiency)
        
        # Calculate summary statistics
        results['avg_cpu_utilization'] = np.mean(results['cpu_utilizations'])
        results['avg_ram_utilization'] = np.mean(results['ram_utilizations'])
        results['avg_wasted_cpu'] = np.mean(results['wasted_cpu'])
        results['avg_wasted_ram'] = np.mean(results['wasted_ram'])
        results['total_cost'] = np.sum(results['costs'])
        results['avg_efficiency'] = np.mean(results['efficiency_scores'])
        results['sla_violation_rate'] = (results['sla_violations'] / max(len(results['cpu_utilizations']), 1)) * 100
        
        return results
    
    def simulate_threshold_scaling(self, data, scale_up_threshold=None, scale_down_threshold=0.40, sla_threshold=0.9):
        """
        Simulate threshold-based scaling (non-AI simple reactive logic).
        """
        results = {
            'cpu_utilizations': [],
            'ram_utilizations': [],
            'wasted_cpu': [],
            'wasted_ram': [],
            'costs': [],
            'efficiency_scores': [],
            'allocations': [],
            'sla_violations': 0
        }
        
        # Use autoscaler defaults if not provided
        if scale_up_threshold is None:
            scale_up_threshold = self.autoscaler.scale_up_threshold
            
        # Start with 1 instance
        current_instances = 1
        base_cpu_unit = 100.0
        base_ram_unit = 100.0
        
        for _, row in data.iterrows():
            # Get current demand
            cpu_demand = row['cpu_usage']
            ram_demand = row['ram_usage']
            
            # 1. Calculate current utilization (before scaling)
            current_capacity_cpu = current_instances * base_cpu_unit
            current_capacity_ram = current_instances * base_ram_unit
            
            util_cpu = cpu_demand / current_capacity_cpu
            util_ram = ram_demand / current_capacity_ram
            max_util = max(util_cpu, util_ram)
            
            # 2. Apply scaling rules for NEXT step (or immediate reaction depending on model)
            # Here we assume reactive: capacity adjusts for current load if it violates constraints,
            # or we simulate the "reaction" to the previous step. 
            # For simplicity and fairness with AI (which predicts ahead), 
            # we'll implement this as "Reactive": Check utilization -> Scale -> Recalculate cost
            
            # Simple Threshold Logic
            if max_util > scale_up_threshold:
                # Scale up
                current_instances = min(self.autoscaler.max_instances, current_instances + 1)
            elif max_util < scale_down_threshold and current_instances > self.autoscaler.min_instances:
                # Scale down
                current_instances = max(self.autoscaler.min_instances, current_instances - 1)
            
            # 3. Calculate metrics with (potentially new) instance count
            final_capacity_cpu = current_instances * base_cpu_unit
            final_capacity_ram = current_instances * base_ram_unit
            
            final_util_cpu = min(1.0, max(0.0, cpu_demand / final_capacity_cpu))
            final_util_ram = min(1.0, max(0.0, ram_demand / final_capacity_ram))
            
            wasted_cpu = max(0, final_capacity_cpu - cpu_demand)
            wasted_ram = max(0, final_capacity_ram - ram_demand)
            
            cost = self.optimizer.calculate_cost(
                allocated_cpu=final_capacity_cpu,
                allocated_ram=final_capacity_ram,
                used_cpu=cpu_demand,
                used_ram=ram_demand,
                instance_count=current_instances
            )
            
            avg_util = (final_util_cpu + final_util_ram) / 2
            cost_eff = 1.0 / (1.0 + cost / 100.0)
            efficiency = (avg_util * 0.7 + cost_eff * 0.3)
            
            # Track SLA violation
            if final_util_cpu > sla_threshold or final_util_ram > sla_threshold:
                results['sla_violations'] += 1
                
            results['cpu_utilizations'].append(final_util_cpu)
            results['ram_utilizations'].append(final_util_ram)
            results['wasted_cpu'].append(wasted_cpu)
            results['wasted_ram'].append(wasted_ram)
            results['costs'].append(cost)
            results['efficiency_scores'].append(efficiency)
            results['allocations'].append({
                'cpu': final_capacity_cpu,
                'ram': final_capacity_ram,
                'instances': current_instances
            })
            
        # Calculate summary statistics
        results['avg_cpu_utilization'] = np.mean(results['cpu_utilizations'])
        results['avg_ram_utilization'] = np.mean(results['ram_utilizations'])
        results['avg_wasted_cpu'] = np.mean(results['wasted_cpu'])
        results['avg_wasted_ram'] = np.mean(results['wasted_ram'])
        results['total_cost'] = np.sum(results['costs'])
        results['avg_efficiency'] = np.mean(results['efficiency_scores'])
        results['sla_violation_rate'] = (results['sla_violations'] / max(len(results['cpu_utilizations']), 1)) * 100
        
        return results
    
    def simulate_ai_optimized(self, data, predictor, preprocessor,
                              lookback_window=24, decision_logger=None, sla_threshold=0.9):
        """Predict demand, allocate CPU/RAM elastically, and size instance count separately.

        Per interval, the optimizer requests pooled elastic CPU/RAM allocations.
        The autoscaler independently sizes the instance fleet. Effective
        provisioned allocations are capped by the fleet capacity, so cooldowns
        and maximum-instance limits cannot create unprovisioned resources.
        """
        results = {
            'cpu_utilizations': [], 'ram_utilizations': [],
            'wasted_cpu': [], 'wasted_ram': [], 'costs': [],
            'efficiency_scores': [], 'allocations': [],
            'sla_violations': 0, 'prediction_failures': 0,
            'scale_up_events': 0, 'scale_down_events': 0,
            'capacity_shortfall_steps': 0, 'requested_cpu': [], 'requested_ram': [],
        }
        current_instances = self.autoscaler.min_instances
        self.autoscaler.reset()

        for i in range(lookback_window, len(data)):
            recent_data = data.iloc[i-lookback_window:i]
            actual_cpu = float(data.iloc[i]['cpu_usage'])
            actual_ram = float(data.iloc[i]['ram_usage'])
            prediction_ok = True
            try:
                prediction = predictor.predict_next_step(recent_data, preprocessor)
                predicted_cpu = float(prediction['cpu_usage'])
                predicted_ram = float(prediction['ram_usage'])
            except Exception as exc:
                prediction_ok = False
                results['prediction_failures'] += 1
                print(f'Prediction fallback at evaluation row {i}: {exc}')
                # Explicit persistence fallback: last observed workload values.
                predicted_cpu = float(recent_data.iloc[-1]['cpu_usage'])
                predicted_ram = float(recent_data.iloc[-1]['ram_usage'])

            allocation = self.optimizer.allocate_elastic_resources(predicted_cpu, predicted_ram)
            scaling = self.autoscaler.plan_instance_count(
                allocation['allocated_cpu'], allocation['allocated_ram'], current_instances)
            current_instances = scaling['new_instances']
            if scaling['action'] == 'SCALE_UP':
                results['scale_up_events'] += 1
            elif scaling['action'] == 'SCALE_DOWN':
                results['scale_down_events'] += 1

            requested_cpu = allocation['allocated_cpu']
            requested_ram = allocation['allocated_ram']
            # The fleet is a hard capacity constraint. Elastic allocations can
            # use less than fleet capacity, but can never exceed it. During a
            # cooldown (or at max_instances), unmet allocation remains visible
            # as higher utilization/SLA risk rather than fictitious capacity.
            fleet_cpu_capacity = (current_instances * self.autoscaler.cpu_capacity_per_instance)
            fleet_ram_capacity = (current_instances * self.autoscaler.ram_capacity_per_instance)
            allocated_cpu = min(requested_cpu, fleet_cpu_capacity)
            allocated_ram = min(requested_ram, fleet_ram_capacity)
            results['requested_cpu'].append(requested_cpu)
            results['requested_ram'].append(requested_ram)
            if actual_cpu > allocated_cpu or actual_ram > allocated_ram:
                results['capacity_shortfall_steps'] += 1

            cpu_util = actual_cpu / allocated_cpu if allocated_cpu > 0 else float('inf')
            ram_util = actual_ram / allocated_ram if allocated_ram > 0 else float('inf')
            # Keep raw utilization for SLA detection; cap only displayed values.
            if cpu_util > sla_threshold or ram_util > sla_threshold:
                results['sla_violations'] += 1
            cpu_util_display = min(1.0, max(0.0, cpu_util))
            ram_util_display = min(1.0, max(0.0, ram_util))

            # Cost only the resources actually provisioned, plus the independent
            # instance base charge. Do not bill the unfulfilled request.
            cost = self.optimizer.calculate_cost(
                allocated_cpu, allocated_ram, actual_cpu, actual_ram,
                instance_count=current_instances)
            avg_util = (cpu_util_display + ram_util_display) / 2.0
            cost_eff = 1.0 / (1.0 + cost / 100.0)
            efficiency = avg_util * 0.7 + cost_eff * 0.3

            results['cpu_utilizations'].append(cpu_util_display)
            results['ram_utilizations'].append(ram_util_display)
            results['wasted_cpu'].append(max(0.0, allocated_cpu - actual_cpu))
            results['wasted_ram'].append(max(0.0, allocated_ram - actual_ram))
            results['costs'].append(cost)
            results['efficiency_scores'].append(efficiency)
            results['allocations'].append({
                'cpu': allocated_cpu, 'ram': allocated_ram,
                'requested_cpu': requested_cpu, 'requested_ram': requested_ram,
                'fleet_cpu_capacity': fleet_cpu_capacity,
                'fleet_ram_capacity': fleet_ram_capacity,
                'instances': current_instances,
                'predicted_cpu': predicted_cpu, 'predicted_ram': predicted_ram,
                'actual_cpu': actual_cpu, 'actual_ram': actual_ram,
                'prediction_ok': prediction_ok, 'scaling_action': scaling['action'],
            })

            if decision_logger is not None:
                timestamp = data.iloc[i]['timestamp'] if 'timestamp' in data.columns else None
                decision_logger.log_decision(
                    step=i - lookback_window + 1,
                    predicted_cpu=predicted_cpu, predicted_ram=predicted_ram,
                    sla_threshold=sla_threshold,
                    decision_dict={**scaling, 'reason': scaling['reason']},
                    timestamp=timestamp)

        n = max(1, len(results['cpu_utilizations']))
        results['avg_cpu_utilization'] = float(np.mean(results['cpu_utilizations']))
        results['avg_ram_utilization'] = float(np.mean(results['ram_utilizations']))
        results['avg_wasted_cpu'] = float(np.mean(results['wasted_cpu']))
        results['avg_wasted_ram'] = float(np.mean(results['wasted_ram']))
        results['total_cost'] = float(np.sum(results['costs']))
        results['avg_efficiency'] = float(np.mean(results['efficiency_scores']))
        results['sla_violation_rate'] = results['sla_violations'] / n * 100.0
        results['avg_instance_count'] = float(np.mean([a['instances'] for a in results['allocations']])) if results['allocations'] else 0.0
        results['max_instance_count'] = max([a['instances'] for a in results['allocations']], default=0)
        results['capacity_shortfall_rate'] = results['capacity_shortfall_steps'] / n * 100.0
        results['avg_requested_cpu'] = float(np.mean(results['requested_cpu'])) if results['requested_cpu'] else 0.0
        results['avg_requested_ram'] = float(np.mean(results['requested_ram'])) if results['requested_ram'] else 0.0
        return results

    def compare_strategies(self, baseline_results, ai_results):
        """
        Compare baseline and AI-optimized strategies.
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        
        Returns:
        --------
        dict : Comparison metrics
        """
        comparison = {
            'cpu_utilization_improvement': (
                (ai_results['avg_cpu_utilization'] - baseline_results['avg_cpu_utilization']) /
                max(baseline_results['avg_cpu_utilization'], 0.01) * 100
            ),
            'ram_utilization_improvement': (
                (ai_results['avg_ram_utilization'] - baseline_results['avg_ram_utilization']) /
                max(baseline_results['avg_ram_utilization'], 0.01) * 100
            ),
            'wasted_cpu_reduction': (
                (baseline_results['avg_wasted_cpu'] - ai_results['avg_wasted_cpu']) /
                max(baseline_results['avg_wasted_cpu'], 0.01) * 100
            ),
            'wasted_ram_reduction': (
                (baseline_results['avg_wasted_ram'] - ai_results['avg_wasted_ram']) /
                max(baseline_results['avg_wasted_ram'], 0.01) * 100
            ),
            'cost_reduction': (
                (baseline_results['total_cost'] - ai_results['total_cost']) /
                max(baseline_results['total_cost'], 0.01) * 100
            ),
            'efficiency_improvement': (
                (ai_results['avg_efficiency'] - baseline_results['avg_efficiency']) /
                max(baseline_results['avg_efficiency'], 0.01) * 100
            )
        }
        
        return comparison
    
    def generate_report(self, baseline_results, ai_results, comparison):
        """
        Generate a text report of simulation results.
        
        Parameters:
        -----------
        baseline_results : dict
            Baseline simulation results
        ai_results : dict
            AI-optimized simulation results
        comparison : dict
            Comparison metrics
        
        Returns:
        --------
        str : Formatted report
        """
        report = "\n" + "="*70 + "\n"
        report += "CLOUD RESOURCE OPTIMIZATION - EXPERIMENTAL EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        report += "📊 BASELINE STRATEGY (Fixed Allocation)\n"
        report += "-"*70 + "\n"
        report += f"Average CPU Utilization: {baseline_results['avg_cpu_utilization']*100:.2f}%\n"
        report += f"Average RAM Utilization: {baseline_results['avg_ram_utilization']*100:.2f}%\n"
        report += f"Average Wasted CPU: {baseline_results['avg_wasted_cpu']:.2f} units\n"
        report += f"Average Wasted RAM: {baseline_results['avg_wasted_ram']:.2f} units\n"
        formatted_baseline_cost = format_indian_currency(baseline_results['total_cost'])
        report += f"Total Cost: Rs {formatted_baseline_cost}\n"
        report += f"Average Efficiency Score: {baseline_results['avg_efficiency']:.4f}\n"
        report += f"SLA Violation Rate: {baseline_results.get('sla_violation_rate', 0):.2f}%\n\n"
        
        report += "🤖 AI-OPTIMIZED STRATEGY (Dynamic Allocation)\n"
        report += "-"*70 + "\n"
        report += f"Average CPU Utilization: {ai_results['avg_cpu_utilization']*100:.2f}%\n"
        report += f"Average RAM Utilization: {ai_results['avg_ram_utilization']*100:.2f}%\n"
        report += f"Average Wasted CPU: {ai_results['avg_wasted_cpu']:.2f} units\n"
        report += f"Average Wasted RAM: {ai_results['avg_wasted_ram']:.2f} units\n"
        formatted_ai_cost = format_indian_currency(ai_results['total_cost'])
        report += f"Total Cost: Rs {formatted_ai_cost}\n"
        report += f"Average Efficiency Score: {ai_results['avg_efficiency']:.4f}\n"
        report += f"SLA Violation Rate: {ai_results.get('sla_violation_rate', 0):.2f}%\n\n"
        
        report += "📈 IMPROVEMENT METRICS\n"
        report += "-"*70 + "\n"
        report += f"CPU Utilization Improvement: {comparison['cpu_utilization_improvement']:+.2f}%\n"
        report += f"RAM Utilization Improvement: {comparison['ram_utilization_improvement']:+.2f}%\n"
        report += f"Wasted CPU Reduction: {comparison['wasted_cpu_reduction']:+.2f}%\n"
        report += f"Wasted RAM Reduction: {comparison['wasted_ram_reduction']:+.2f}%\n"
        report += f"Cost Reduction: {comparison['cost_reduction']:+.2f}%\n"
        report += f"Efficiency Improvement: {comparison['efficiency_improvement']:+.2f}%\n"
        report += "="*70 + "\n"
        
        return report


if __name__ == "__main__":
    # Test cloud simulator
    print("🔄 Testing cloud simulator...")
    
    simulator = CloudSimulator()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='15min'),
        'cpu_usage': np.random.uniform(20, 80, 100),
        'ram_usage': np.random.uniform(25, 75, 100),
        'requests': np.random.randint(50, 500, 100)
    })
    
    # Simulate baseline
    baseline = simulator.simulate_baseline(sample_data, fixed_cpu=100, fixed_ram=100)
    
    print("\n📊 Baseline Results:")
    print(f"   Avg CPU Utilization: {baseline['avg_cpu_utilization']*100:.2f}%")
    print(f"   Avg RAM Utilization: {baseline['avg_ram_utilization']*100:.2f}%")
    formatted_cost = format_indian_currency(baseline['total_cost'])
    print(f"   Total Cost: Rs {formatted_cost}")
    
    print("\n✅ Cloud simulator test completed!")
    print("⚠️  Note: Full AI simulation requires trained model.")


