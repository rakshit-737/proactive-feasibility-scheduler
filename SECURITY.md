# Security Policy

## Scope

This is a **research codebase** for simulation-driven job-scheduling experiments. It is
not a production service and ships no network endpoints, authentication, or user data
handling. The practical risk surface is therefore small, but reports are still welcome.

## Reporting a vulnerability

Please report suspected security issues privately rather than opening a public issue —
either through a [GitHub security advisory](https://github.com/rakshit-737/proactive-feasibility-scheduler/security/advisories/new)
or by email to rakshitoffl@gmail.com. Include reproduction steps and impact. You can
expect an acknowledgement within a few days.

## Notes for users running this code

- The pipeline executes local Python scripts and reads workload traces; run it in an
  environment you trust.
- Real workload traces (`.swf` / `.swf.gz`) are third-party data from the
  [Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/); obtain
  them from their original source and review their terms before redistribution.
- Model files (`*.pkl`) are Python pickles. Only load pickles you have generated yourself
  or obtained from a source you trust, since unpickling can execute arbitrary code.
