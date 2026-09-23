# JIGSAWS complete-trial visual-attention Skill

## Scope

Judge one complete robotic suturing trial on the six released modified Global
Rating Scale items. The 18 images are sparse chronological observations: 12
provide uniform coverage and 6 are placed at temporal review pointers. The
motion plugin is advisory and cannot issue a score.

Use the common 1--5 scale: 1 means clear and frequent serious problems; 2 means
important recurring problems; 3 means mixed or adequate execution; 4 means
consistently proficient execution supported at multiple points; and 5 means
exceptional, near-flawless execution with strong repeated support.

## Binding evidence-authority rules

1. Follow `rubric_evidence_scope`. Only `time_and_motion` has direct numeric
   support. Every other item requires visible evidence in the supplied images.
2. A temporal review pointer only says where to inspect an image. Its named
   action is neither an error nor evidence of proficiency; score direction must
   come from what is visibly happening in that image.
3. A low near-stationary fraction means fewer near-stationary frames, not more
   pauses. Interpret duration, travel, jerk, reversals, and stationary time
   together; no single measurement is a verdict.
4. Missing detections and absence of obvious errors in sparse images are not
   positive evidence. Prefer score 3 over unsupported 4 or 5.
5. Cite concrete visible behavior for every score above or below 3. If sparse
   images cannot show direction, assign 3 and state the limitation.

## Items

1. **respect_for_tissue**: require visible deformation, rough contact, repeated
   manipulation, or another direct visual sign for a directional judgment.
2. **suture_needle_handling**: judge secure control, orientation, loss,
   entanglement, repeated corrective regrasp, and recovery from the images.
3. **time_and_motion**: use the complete-trial kinematic measurements to assess
   economy while respecting every measurement's stated direction and limits.
4. **flow_of_operation**: require visible corroboration of interruption,
   recovery, unnecessary repetition, or uncertain action.
5. **overall_performance**: integrate only supported item evidence.
6. **quality_of_final_product**: use late images showing placement, form,
   spacing, and consistency. Motion measurements cannot establish product
   quality.

The frozen foundation MLLM owns every score. Never use expert labels,
self-reported experience, operator identity, or ground-truth transcripts.
