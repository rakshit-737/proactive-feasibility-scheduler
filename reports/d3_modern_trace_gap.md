# D3 — a modern GPU-cluster trace: what is needed, and why it is not here

**Status: NOT DONE. This is a gap, deliberately left open rather than faked.**

The external-validity story currently rests on two real machines, both from the 1990s:
LANL CM-5 (1994) and SDSC SP2 (1998). That is the most attackable part of the paper,
and a third substrate from a modern GPU cluster would materially strengthen it —
the degeneracy argument is about feature-set structure, not about hardware, so it
should hold on a 2020s trace too, and showing that it does would be worth more than
any amount of additional synthetic evidence.

## Why it is absent

No modern GPU-cluster trace is in this repository, and none was added. The candidate
traces are multi-gigabyte downloads behind registration, click-through licences, or
institutional agreements, and this environment has no practical route to any of them.

This file exists because the alternative was worse. Version 3.6 of this repository
deleted a code path that **synthesised a wait-time target from a hard-coded linear
formula while tagging its output `source_type="real"`**, along with scaffolding for an
Alibaba trace that had never existed here — `ALIBABA_COLUMNS`, a `_map_alibaba`
mapper, an expected path, and download instructions, all for a file nothing had ever
read. A project that has already shipped a fake trace once does not get to ship a
second one. An honest empty slot beats a plausible substitute.

## What a candidate must supply

The degeneracy diagnostic needs, per job:

| quantity | why | SWF field |
|---|---|---|
| submit time | defines the dispatch instant | 2 |
| wait time | the prediction target | 3 |
| run time | cluster-state reconstruction | 4 |
| requested size | **the variable the whole claim is about** | 8 |
| capacity | to replay occupancy | header `MaxProcs` |

and, for the Phase D non-degeneracy sweep, at least one genuine per-job attribute:
requested time (field 9), user id (12), or queue (15).

A trace missing *requested size* cannot be used at all. A trace missing *wait time*
can still support the degeneracy diagnostic — which only needs the model's scores at
each instant — but not the scheduler comparison.

## Candidates, with the specific obstacle

| trace | era | obstacle |
|---|---|---|
| Alibaba PAI / cluster-trace-v2018 | 2018–2020 | not SWF; needs a converter. GPU sub-trace records container scheduling, and the mapping to "requested size" is a modelling decision that must be defended, not assumed |
| Microsoft Philly | 2017–2018 | released as JSON job logs; per-job requested GPU count present. Needs a converter and a licence check |
| HKUST Helios | 2020 | SWF-like; the closest fit to the existing harness |
| Google Borg 2019 | 2019 | enormous; not GPU-specific; requested resources are normalised, so "size" needs a defensible definition |
| SenseTime Acme | 2023 | most modern; availability is the constraint |

## The honest procedure, when a trace is obtained

1. Add it as a compressed file beside the existing two, read through
   `02_data/swf_io.open_swf` (which already handles both plain and gzipped forms).
2. Extend `TRACES` in `04_scheduler/trace_driven_benchmark.py`. Do **not** add a
   mapper that invents any field the trace does not carry. If a needed field is
   missing, say exactly which one and stop — that is what the removed code failed to do.
3. Re-run the degeneracy diagnostic. The prediction is **zero violations**, because
   Proposition 1 in `METHODOLOGY.md` is about the feature map, not the machine.
4. Re-run the non-degeneracy sweep if the trace carries a per-job attribute. The
   prediction is that violations appear exactly when such an attribute is added.
5. If the diagnostic finds violations on the baseline feature set, that is a genuine
   counterexample and the paper's central claim needs qualifying. Report it.

## What the paper must say until then

That the result is established on two real machines, both from the 1990s, plus a
synthetic generator; that the argument is structural and predicts the same outcome on
modern hardware; and that this prediction is **untested**. The trace vintage is
already listed in the manuscript's Threats to Validity section, and it should stay
there, stated in exactly those terms.
