# An Evidence-Based Framework for Procedural Video Assessment

A framework connecting SOP criteria, specialist observations,
temporal evidence and multimodal model judgments.

## Components

- `cvs_assessment/`: evidence contracts, visual adapters, temporal graphs,
  task packages, judgment policies, model definitions and metric computation.
- `procedural_assessment/`: compatibility API and task registration.
- `scripts/`: training and inference runners.
- `specs/` and `configs/`: reusable skill definitions and generic contracts.
- `tool_augment/`: geometry, wiping trajectory and hand-pose algorithms,
  validated model requests, tool dispatch and evidence return.
- `tests/`: synthetic contract and algorithm tests.

## Setup

Install the relevant requirements in a suitable Python environment. Training
and model serving may require separate environments. Configure the dataset paths,
model weights, task specifications and model endpoint for the selected runner.

## Entry points

- Generic assessment: `scripts/run_assessment.py`.
- Component-state judgment: `scripts/run_industreal_consensus_residual_arbitration.py`.
- Phase judgment: `scripts/run_cholect50_qwen_phase_pair.py`.
- Graph memory: `scripts/build_cholec_graph_memory.py` and `scripts/run_cholec_graph_main.py`.
- Model service: `scripts/serve_qwen3_vl_openai.py`.
- Tool workflow: see `tool_augment/README.md`.

## Tests

```bash
python -m unittest discover -s tests
python -m unittest discover -s tool_augment -p 'test_hand_press.py'
python -m unittest discover -s tool_augment/constrained_assessment -p 'test_contracts.py'
```
