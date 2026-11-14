# Cloud Resource Optimization — V4 patch

## Chosen infrastructure model
V4 models **elastic pooled CPU and RAM allocations independently** from the **number of running instances**. The optimizer calculates the required CPU and RAM quantities from predicted demand. The autoscaler then sizes the instance fleet using separately configured per-instance hosting limits (default: 100 CPU units and 100 RAM units per instance).

The per-instance limits are a fleet-sizing assumption, not a claim that CPU and RAM allocations are fixed-size VM resources. The CPU/RAM quantities remain independently elastic. Instance count affects the base instance charge only; it does not multiply the pooled CPU/RAM allocation a second time.

## Files to replace
- `optimization_engine.py`
- `autoscaler.py`
- `cloud_simulator.py`
- `main.py`

Back up these four files first. Extract the patch into the project root and overwrite them. Other project files are not included and should remain unchanged.

## V4 behaviour
- Independent CPU and RAM elastic allocation with a target utilization and safety buffer.
- Instance count derived separately from allocated CPU/RAM footprint.
- One-interval scale-up cooldown and three-interval scale-down cooldown by default.
- Prediction failure uses a clearly counted persistence fallback (last observed CPU/RAM), not an unlabelled optimizer fallback.
- AI utilization, waste, SLA and cost use actual observed demand against the same elastic allocation.
- AI efficiency is recalculated from actual utilization and actual simulated cost.
- Per-scenario report includes average/max AI instance count, scale events and prediction fallbacks.
- Baseline fixed allocation cost assumes one 100x100 capacity instance, correcting the previous inferred instance count.

## Important interpretation
The cost parameters remain illustrative simulation parameters inherited from V3. They are not AWS/Azure/GCP rates. The per-instance 100 CPU / 100 RAM hosting limits are also modelling assumptions. Do not describe output as real-world cloud savings without calibration and sensitivity analysis.

The existing threshold strategy remains reactive and uses its existing 100-unit-per-instance capacity assumption. V4 makes the AI strategy's elastic allocation and instance accounting internally consistent, but this is not yet a perfect real-world apples-to-apples cloud benchmark: provisioning delay, billing granularity, placement constraints, and capacity fragmentation are not modelled.

## Run
From the project root with the virtual environment active:

```powershell
python -m compileall -q .
python main.py
```

Expected generated files include `outputs/scenario_results.txt`, `outputs/final_experiment_summary.txt`, and the existing plots/decision log. Keep the V3 outputs separately for version-to-version comparison.
