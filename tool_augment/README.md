# Tool-augmented assessment

The model emits a structured request, a dispatcher validates it against the
criterion's tool rules, specialist algorithms return observations, and the model
judges the criterion using the returned evidence. Tools do not supply clinical
compliance labels.

## Source

- `reproduce.py`: headless specialist adapters; pass a module name, `--source` and `--output`.
- `hand_press.py`: hand landmarks, pose features and temporal support.
- `constrained_assessment/run.py`: request validation, dispatch and evidence return.
- `constrained_assessment/evidence_views.py`: geometric and trajectory measurements.
- `constrained_assessment/verify_run.py`: mechanical consistency checks.
- `extracted/tool_augment_code/src/`: specialist detection and training algorithms.

Install the supplemental requirements plus PyTorch, Ultralytics, Pillow,
matplotlib and jsonschema as needed. Place detector weights under
`extracted/tool_augment_code/models/`.

Supply your own local JSON manifest as a list of objects with `case_id`
(`case_<digits>`), `criterion_id` and an absolute `source` video path.
Set `SOP_CONFIG` to your SOP JSON;
otherwise the runner uses `constrained_assessment/sops.example.json`.
The example settings are illustrative, not validated application parameters.

```bash
SOP_CONFIG=/path/to/sops.json python tool_augment/constrained_assessment/run.py prepare --cases /path/to/cases.json --output /path/to/output
SOP_CONFIG=/path/to/sops.json python tool_augment/constrained_assessment/run.py run --output /path/to/output --base-url http://127.0.0.1:8000/v1
```

Use the same SOP configuration for preparation and execution.
