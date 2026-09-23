# JIGSAWS suturing skill-assessment Skill

## Decision scope

Assess the quality of a complete robotic suturing trial from the supplied video
frames and advisory motion facts. The final target is the expert modified Global
Rating Scale, not the operator's self-reported experience level and not gesture
classification.

For every shown timestamp, report the quality evidence visible at that point.
The trial-level evaluator will aggregate the time-indexed judgments. Do not copy
one early judgment to every timestamp. If a criterion cannot yet be seen, use
`U`; in particular, the final product is usually not assessable until late in
the trial.

Use these framework states consistently:

- `N`: clearly poor execution, corresponding to expert ratings 1--2;
- `P`: intermediate or mixed execution, corresponding to rating 3;
- `F`: clearly proficient execution, corresponding to ratings 4--5;
- `U`: the available images and facts do not support a judgment.

## Criteria

1. **respect_for_tissue**: look for controlled manipulation without unnecessary
   force, repeated contact, or rough handling of the practice material.
2. **suture_needle_handling**: look for secure, deliberate needle and suture
   control, suitable orientation, and limited loss or recovery.
3. **time_and_motion**: look for purposeful movement with limited avoidable
   travel, hesitation, or idle time.
4. **flow_of_operation**: look for coherent progress with limited interruption,
   repetition, or uncertainty between actions.
5. **overall_performance**: judge the complete visible execution for control,
   efficiency, and independence.
6. **quality_of_final_product**: judge only the visible completed suture for
   placement, form, spacing, and consistency.

Robot-kinematic summaries are fallible observations. Path length, speed, jerk,
gripper activity, or bimanual correlation may explain motion, but no single
number proves skill. The video remains necessary for semantic interpretation
and final-product quality. Missing motion facts are not evidence of failure.

Never use a trial identifier, operator identity, self-reported level, expert
score, or ground-truth gesture transcript as evidence. The frozen foundation
MLLM owns every final state; evidence plugins do not issue a skill verdict.

