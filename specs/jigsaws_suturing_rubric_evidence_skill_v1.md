# JIGSAWS complete-trial rubric-evidence Skill

## Scope

Judge one complete robotic suturing trial on the six released modified Global
Rating Scale items. The images are sparse chronological observations. The
rubric-aligned motion plugin is advisory and cannot issue a score.

Use the common 1--5 scale: 1 means clear and frequent serious problems; 2 means
important recurring problems; 3 means mixed or adequate execution; 4 means
consistently proficient execution supported at multiple points; and 5 means
exceptional, near-flawless execution with strong repeated support.

## Binding evidence-authority rules

1. Follow `rubric_evidence_scope`. Motion and predicted gesture facts alone
   cannot raise or lower `respect_for_tissue` or `quality_of_final_product`.
2. Gesture-model confidence affects `evidence_adequacy` only. It must never
   raise or lower any GRS score.
3. Primitive counts, durations, ratios, A-B-A revisits, and attention intervals
   have no validated good/bad direction. Use them only to locate images for
   review. They may affect score direction only when the cited images show a
   concrete handling error, recovery, interruption, or repetition.
4. A low near-stationary fraction means fewer near-stationary frames, not more
   pauses. Interpret duration, travel, jerk, reversals, and stationary time
   together; no single measurement is a verdict.
5. Missing detections and absence of obvious errors in sparse images are not
   positive evidence. Prefer score 3 over unsupported 4 or 5.

## Items

1. **respect_for_tissue**: require visible deformation, rough contact, repeated
   manipulation, or other direct visual evidence for a directional judgment.
   The supplied plugin has no force or deformation measurement.
2. **suture_needle_handling**: judge secure control, orientation, loss,
   entanglement, repeated corrective regrasp, and recovery from the images.
   Predicted gesture identity and primitive ratios can only locate review times.
3. **time_and_motion**: use the complete-trial kinematic measurements to assess
   economy, while respecting each measurement's stated direction and limits.
4. **flow_of_operation**: require visible corroboration of interruption,
   recovery, unnecessary repetition, or uncertain transitions. A-B-A patterns
   can arise from the normal repeated four-pass structure.
5. **overall_performance**: integrate the supported item evidence; do not use
   operator identity, experience, or small-model confidence.
6. **quality_of_final_product**: use only late images showing placement, form,
   spacing, and consistency. Motion and gesture facts cannot establish product
   quality.

The frozen foundation MLLM owns every score. Never use expert labels,
self-reported experience, operator identity, or ground-truth transcripts.
