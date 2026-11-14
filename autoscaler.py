"""Auto-scaling decisions — V4.

Instance count is a separate infrastructure dimension. CPU and RAM quantities
remain elastic pooled allocations and are not multiplied by instance count.
"""
import math


class AutoScaler:
    def __init__(self, scale_up_threshold=0.75, scale_down_threshold=0.40,
                 min_instances=1, max_instances=10, scale_factor=1.5,
                 cpu_capacity_per_instance=100.0, ram_capacity_per_instance=100.0,
                 scale_up_cooldown=1, scale_down_cooldown=3):
        if not 0 <= scale_down_threshold < scale_up_threshold <= 1:
            raise ValueError('Thresholds must satisfy 0 <= scale-down < scale-up <= 1.')
        self.scale_up_threshold = float(scale_up_threshold)
        self.scale_down_threshold = float(scale_down_threshold)
        self.min_instances = int(min_instances)
        self.max_instances = int(max_instances)
        self.scale_factor = float(scale_factor)
        self.cpu_capacity_per_instance = float(cpu_capacity_per_instance)
        self.ram_capacity_per_instance = float(ram_capacity_per_instance)
        self.scale_up_cooldown = max(0, int(scale_up_cooldown))
        self.scale_down_cooldown = max(0, int(scale_down_cooldown))
        self.reset()

    def reset(self):
        self._steps_since_scale = max(self.scale_up_cooldown, self.scale_down_cooldown)

    def plan_instance_count(self, allocated_cpu, allocated_ram, current_instances=1):
        """Size instance fleet to elastic resource footprint; count stays separate."""
        current = max(self.min_instances, min(self.max_instances, int(current_instances)))
        cpu_nodes = math.ceil(max(0.0, float(allocated_cpu)) / self.cpu_capacity_per_instance)
        ram_nodes = math.ceil(max(0.0, float(allocated_ram)) / self.ram_capacity_per_instance)
        required = max(self.min_instances, cpu_nodes, ram_nodes)
        required = min(self.max_instances, required)
        if required > current:
            if self._steps_since_scale >= self.scale_up_cooldown:
                new_count, action = required, 'SCALE_UP'
            else:
                new_count, action = current, 'COOLDOWN'
        elif required < current:
            if self._steps_since_scale >= self.scale_down_cooldown:
                new_count, action = required, 'SCALE_DOWN'
            else:
                new_count, action = current, 'COOLDOWN'
        else:
            new_count, action = current, 'NO_ACTION'
        if new_count != current:
            self._steps_since_scale = 0
        else:
            self._steps_since_scale += 1
        return {
            'action': action, 'current_instances': current,
            'new_instances': new_count, 'required_instances': required,
            'cpu_capacity_per_instance': self.cpu_capacity_per_instance,
            'ram_capacity_per_instance': self.ram_capacity_per_instance,
            'reason': f'Elastic allocation requires {required} instance(s); fleet is {new_count}.'
        }

    def decide_scaling_action(self, predicted_cpu, predicted_ram,
                              current_cpu_allocation, current_ram_allocation,
                              current_instances=1):
        """Legacy threshold-based decision API retained for V3 compatibility."""
        total_cpu = max(0.0, current_cpu_allocation * current_instances)
        total_ram = max(0.0, current_ram_allocation * current_instances)
        cpu_util = float(predicted_cpu) / total_cpu if total_cpu else 0.0
        ram_util = float(predicted_ram) / total_ram if total_ram else 0.0
        peak = max(cpu_util, ram_util)
        new_count = int(current_instances)
        action = 'NO_ACTION'
        if peak > self.scale_up_threshold:
            new_count = min(self.max_instances, max(new_count + 1, math.ceil(new_count * self.scale_factor)))
            action = 'SCALE_UP' if new_count != current_instances else 'AT_MAXIMUM'
        elif peak < self.scale_down_threshold and new_count > self.min_instances:
            new_count = max(self.min_instances, min(new_count - 1, math.floor(new_count / self.scale_factor)))
            action = 'SCALE_DOWN'
        return {
            'action': action, 'new_instances': new_count,
            'current_instances': int(current_instances),
            'reason': f'Peak utilization {peak:.3f}; action {action}.',
            'cpu_utilization': min(1.0, max(0.0, cpu_util)),
            'ram_utilization': min(1.0, max(0.0, ram_util)),
            'max_utilization': min(1.0, max(0.0, peak)),
            'predicted_cpu': predicted_cpu, 'predicted_ram': predicted_ram,
        }

    def calculate_resource_allocation(self, total_cpu_needed, total_ram_needed,
                                      num_instances, cpu_per_instance=None,
                                      ram_per_instance=None):
        count = max(1, int(num_instances))
        cpu_each = total_cpu_needed / count if cpu_per_instance is None else cpu_per_instance
        ram_each = total_ram_needed / count if ram_per_instance is None else ram_per_instance
        return {'cpu_per_instance': cpu_each, 'ram_per_instance': ram_each,
                'total_cpu': cpu_each * count, 'total_ram': ram_each * count,
                'num_instances': count}


if __name__ == '__main__':
    scaler = AutoScaler()
    print(scaler.plan_instance_count(165, 72, current_instances=1))
