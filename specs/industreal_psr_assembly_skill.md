# IndustReal assembly procedure assessment Skill

Judge whether each required assembly state has been correctly completed at each shown timestamp.
The task concerns **successful state completion**, not merely seeing a hand motion or an attempted
action.

For an assembly recording, the required milestones are:

1. the front chassis and its front pin are correctly installed;
2. the rear chassis and its two chassis pins are correctly installed;
3. the front bracket and bracket screw are correctly installed;
4. the front wheel assembly is correctly installed;
5. the rear wheel assembly is correctly installed.

For every component criterion:

- use `F` only when the requested installed/removed state is visibly reached with the correct
  part and plausible orientation/connection;
- use `P` when work is underway or some supporting evidence exists but completion is not yet
  established;
- use `N` when the state is not reached, has been undone, or an incorrect part/orientation is
  evident;
- use `U` when the relevant assembly area cannot be judged.

A completed state normally persists until a later removal or correction. Temporal tool events may
help locate a transition, but they are fallible and do not prove correct completion. A detector
`error_state` is a warning to inspect the real image, not an automatic negative verdict. Missing
tool observations are not proof that a component is absent.

Respect procedure order in the case summary and flag omissions, incorrect completions,
out-of-order transitions, and unresolved final errors. The frozen MLLM owns every final state.
