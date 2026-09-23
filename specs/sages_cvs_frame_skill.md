# SAGES Critical View of Safety frame-assessment Skill

## Decision scope

Judge each supplied frame at its own timestamp. The state for one frame is an
instantaneous visual assessment; neighboring frames and temporal-tool facts may
resolve ambiguity, occlusion, or identity, but persistence is not required for
a frame to be marked fully satisfied. Do not use evidence after clipping has
started to justify an earlier frame.

## Criteria

1. **two_structures** is fully satisfied only when exactly the cystic duct and
   cystic artery are visibly distinct and both enter the gallbladder. A tool,
   clip, or unidentified tubular structure is not sufficient.
2. **cystic_plate** is fully satisfied only when the lower third of the
   gallbladder has been dissected away and the cystic plate is visibly exposed.
3. **hepatocystic_triangle** is fully satisfied only when the triangle bounded
   by the cystic duct, common hepatic duct, and liver edge is cleared of fat and
   fibrous tissue. Merely seeing the region or detecting a triangle-shaped ROI
   is not sufficient.

Use partial when supporting anatomy is visible but the complete condition is
not established. Use unknown when visibility is insufficient. Predicted boxes
are fallible positive hints: a missing box never proves that anatomy is absent.

