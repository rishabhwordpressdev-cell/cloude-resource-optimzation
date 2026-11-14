# V5 — Capacity-Aware Predictive Elastic Allocation

## Scope (deadline-focused)
V5 retains the existing LSTM, dataset, workload scenarios, baseline and threshold strategies. It makes a targeted correction to the V4 elastic simulator rather than introducing a new unvalidated prediction model.

## Changes
1. The autoscaler still plans the instance fleet independently from elastic CPU/RAM quantities.
2. Requested elastic CPU/RAM are now capped by the actual fleet capacity (`instance_count × per-instance capacity`).
3. When cooldowns or maximum-instance limits prevent sufficient capacity, the simulator records a capacity shortfall and evaluates actual utilization against provisioned resources. This can increase SLA violations instead of granting fictitious capacity.
4. Cost is calculated from the resources actually provisioned plus the separate instance base charge; unfulfilled requested resources are not charged.
5. Results now include capacity-shortfall rate and requested allocation diagnostics.
6. `main.py` saves three plots for each scenario under `outputs/scenarios/<scenario_name>/`; the legacy top-level plots are explicitly described as the final Gradual Growth scenario.
7. Reports describe INR values as illustrative simulation costs, not real cloud-provider bills.

## Important experimental caveats
- The cost coefficients and billing interval remain illustrative and must be disclosed in the report.
- This is a simulator, not a deployment on AWS/Azure/GCP.
- The source workload is aggregated Google Cluster data. The relative time axis is synthetic because the CSV has no timestamps.
- Keep the V4 source and outputs for comparison. Do not overwrite V4 backups.

## Apply
Back up `main.py` and `cloud_simulator.py`, extract this patch into the project root, then run:

```powershell
python -m compileall -q .
python main.py
```

If the full run cannot finish before submission, use the verified V4 results and label them V4. Do not claim V5 performance without a completed V5 run.
