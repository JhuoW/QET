# Run log

One row per experiment run, appended by `qet.runlog` when a run is submitted and updated when it ends. Agents
may add a note (for example why a run was cancelled) but never type results here: results live in
`results/raw/<run_id>/` and tables come from `scripts/`.

| run_id | submitted | milestone | config | method | backbone | seed | commit | status | GPU-h | path | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
