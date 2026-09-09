# adapter128_scale1p75_chck_70M

returncode: 1
elapsed_sec: 0.1

Traceback (most recent call last):
  File "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py", line 369, in <module>
    main()
  File "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py", line 365, in main
    run_one(args)
  File "experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py", line 331, in run_one
    raise FileNotFoundError(f"Training metrics missing; inspect run before evaluation: {run_dir / 'scientific_metrics.json'}")
FileNotFoundError: Training metrics missing; inspect run before evaluation: experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/scientific_metrics.json

