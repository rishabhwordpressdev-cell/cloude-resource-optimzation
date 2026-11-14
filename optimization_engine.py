"""
Resource Optimization Engine — V4
=================================
Elastic CPU/RAM allocation is calculated independently from VM/instance count.
Workload and allocations use the same normalized resource units as the dataset
adapter (0–100 demand units in the current experiment).
"""


def format_indian_currency(amount):
    amount_int = int(round(amount))
    amount_str = str(abs(amount_int))
    if len(amount_str) <= 3:
        result = amount_str
    else:
        result = amount_str[-3:]
        remaining = amount_str[:-3]
        while remaining:
            result = remaining[-2:] + ',' + result
            remaining = remaining[:-2]
    return ('-' if amount_int < 0 else '') + result


class ResourceOptimizer:
    """Calculate independently elastic CPU and RAM allocations and costs."""

    def __init__(self, cpu_overhead=0.10, ram_overhead=0.10,
                 min_utilization=0.50, max_utilization=0.85,
                 cost_cpu_per_unit=89.83, cost_ram_per_unit=44.92,
                 base_cost_per_instance=20.0, idle_penalty_factor=0.5,
                 safety_buffer=0.10):
        if not 0 < min_utilization < max_utilization <= 1:
            raise ValueError('Utilization bounds must satisfy 0 < min < max <= 1.')
        self.cpu_overhead = float(cpu_overhead)
        self.ram_overhead = float(ram_overhead)
        self.min_utilization = float(min_utilization)
        self.max_utilization = float(max_utilization)
        self.cost_cpu_per_unit = float(cost_cpu_per_unit)
        self.cost_ram_per_unit = float(cost_ram_per_unit)
        self.base_cost_per_instance = float(base_cost_per_instance)
        self.idle_penalty_factor = float(idle_penalty_factor)
        self.safety_buffer = float(safety_buffer)

    @property
    def target_utilization(self):
        return (self.min_utilization + self.max_utilization) / 2.0

    def allocate_elastic_resources(self, predicted_cpu, predicted_ram):
        """Return pooled CPU and RAM allocations independently of instance count."""
        cpu = max(0.1, float(predicted_cpu))
        ram = max(0.1, float(predicted_ram))
        cpu_allocation = cpu / self.target_utilization * (1 + self.cpu_overhead) * (1 + self.safety_buffer)
        ram_allocation = ram / self.target_utilization * (1 + self.ram_overhead) * (1 + self.safety_buffer)
        return {
            'allocated_cpu': round(cpu_allocation, 4),
            'allocated_ram': round(ram_allocation, 4),
            'predicted_cpu': cpu,
            'predicted_ram': ram,
            'target_utilization': self.target_utilization,
        }

    def calculate_cost(self, allocated_cpu, allocated_ram, used_cpu, used_ram, instance_count=1):
        """Illustrative interval cost: elastic resource charges + instance base charge + idle penalty."""
        instance_count = max(1, int(instance_count))
        allocated_cpu = max(0.0, float(allocated_cpu))
        allocated_ram = max(0.0, float(allocated_ram))
        used_cpu = max(0.0, float(used_cpu))
        used_ram = max(0.0, float(used_ram))
        resource_cost = (allocated_cpu * self.cost_cpu_per_unit +
                         allocated_ram * self.cost_ram_per_unit)
        idle_cpu = max(0.0, allocated_cpu - used_cpu)
        idle_ram = max(0.0, allocated_ram - used_ram)
        idle_penalty = (idle_cpu * self.cost_cpu_per_unit +
                        idle_ram * self.cost_ram_per_unit) * self.idle_penalty_factor
        return (instance_count * self.base_cost_per_instance) + resource_cost + idle_penalty

    def optimize_allocation(self, predicted_cpu, predicted_ram,
                            current_cpu=None, current_ram=None, instance_count=1):
        """Backward-compatible interface; instance count is a separate cost input."""
        allocation = self.allocate_elastic_resources(predicted_cpu, predicted_ram)
        cpu_util = min(1.0, max(0.0, float(predicted_cpu) / allocation['allocated_cpu']))
        ram_util = min(1.0, max(0.0, float(predicted_ram) / allocation['allocated_ram']))
        cost = self.calculate_cost(
            allocation['allocated_cpu'], allocation['allocated_ram'],
            predicted_cpu, predicted_ram, instance_count=instance_count)
        utilization_score = (cpu_util + ram_util) / 2
        cost_efficiency = 1.0 / (1.0 + cost / 100.0)
        return {
            'optimal_cpu': allocation['allocated_cpu'],
            'optimal_ram': allocation['allocated_ram'],
            'allocated_cpu': allocation['allocated_cpu'],
            'allocated_ram': allocation['allocated_ram'],
            'cpu_utilization': cpu_util,
            'ram_utilization': ram_util,
            'cost': cost,
            'efficiency_score': utilization_score * 0.7 + cost_efficiency * 0.3,
            'predicted_cpu': float(predicted_cpu),
            'predicted_ram': float(predicted_ram),
        }

    def compare_allocations(self, before_allocation, after_allocation):
        def improvement(before, after):
            return (after - before) / max(abs(before), 0.01) * 100
        before_cpu = max(0, before_allocation['optimal_cpu'] - before_allocation['predicted_cpu'])
        after_cpu = max(0, after_allocation['optimal_cpu'] - after_allocation['predicted_cpu'])
        before_ram = max(0, before_allocation['optimal_ram'] - before_allocation['predicted_ram'])
        after_ram = max(0, after_allocation['optimal_ram'] - after_allocation['predicted_ram'])
        return {
            'cpu_utilization_improvement': improvement(before_allocation['cpu_utilization'], after_allocation['cpu_utilization']),
            'ram_utilization_improvement': improvement(before_allocation['ram_utilization'], after_allocation['ram_utilization']),
            'cost_savings_percent': improvement(before_allocation['cost'], after_allocation['cost']) * -1,
            'efficiency_improvement': improvement(before_allocation['efficiency_score'], after_allocation['efficiency_score']),
            'before_wasted_cpu': before_cpu, 'after_wasted_cpu': after_cpu,
            'before_wasted_ram': before_ram, 'after_wasted_ram': after_ram,
            'wasted_cpu_reduction': improvement(before_cpu, after_cpu) * -1,
            'wasted_ram_reduction': improvement(before_ram, after_ram) * -1,
        }


if __name__ == '__main__':
    engine = ResourceOptimizer()
    result = engine.optimize_allocation(60, 70, instance_count=2)
    print('V4 elastic allocation test')
    print(f"CPU allocation: {result['allocated_cpu']:.2f}")
    print(f"RAM allocation: {result['allocated_ram']:.2f}")
    print(f"Cost with 2 instances: {format_indian_currency(result['cost'])}")
