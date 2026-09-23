# JIGSAWS complete-trial suturing GRS Skill

## Scope

Judge one complete robotic suturing trial on the six released modified Global
Rating Scale items. Produce one 1--5 score per item for the complete trial. Do
not turn individual sampled images into independent skill labels and do not
average repeated copies of the same judgment.

The sampled images are sparse observations. The subject-out motion plugin is
advisory. Its percentiles compare observable motion with other-subject trials;
they are not quality percentiles. A predicted gesture identity is not evidence
of proficiency by itself. Never use operator identity, self-reported experience,
expert scores, or ground-truth gesture transcripts.

## Common scale

- **1**: clear and frequent serious problems for this item;
- **2**: important recurring problems, despite some acceptable execution;
- **3**: mixed or adequate execution without consistent proficiency;
- **4**: consistently proficient execution supported at multiple points or by
  directly relevant complete-trial motion evidence;
- **5**: exceptional, near-flawless execution with strong repeated support.

Do not award 4 or 5 merely because no obvious error appears in sparse frames.
Limited evidence affects evidence adequacy, not the score direction. When the
available evidence is genuinely mixed, prefer 3 over an unsupported extreme.

## Items

1. **respect_for_tissue**: assess controlled interaction with the practice
   material. Repeated forceful deformation, rough contact, or unnecessary
   manipulation supports a lower score. Motion magnitude alone cannot prove
   tissue respect.
2. **suture_needle_handling**: assess secure and deliberate needle/suture
   control, suitable orientation, limited loss, and limited recovery. Gesture
   identity alone cannot prove handling quality.
3. **time_and_motion**: assess economy of movement over the complete trial.
   Duration, path rate, low-motion fraction, repeated segments, and transition
   rate may be relevant only when interpreted together; no single percentile is
   an automatic verdict.
4. **flow_of_operation**: assess coherent progress with limited interruption,
   unnecessary repetition, or uncertainty. A chronological primitive profile
   can support this item, but normal task repetition must not be treated as an
   error without image or motion context.
5. **overall_performance**: integrate control, efficiency, and continuity over
   the complete trial. Do not infer independence from operator identity or
   experience labels.
6. **quality_of_final_product**: judge the completed suture from late frames for
   visible placement, form, spacing, and consistency. Early frames and robot
   motion alone are insufficient for a high or low final-product score.

The frozen foundation MLLM owns every score. The Skill defines how to judge and
the small temporal model supplies observations; neither is a final predictor.
