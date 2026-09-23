# Foundation-centered evidence adjudication policy

## Purpose

Prevent a larger number of fallible plugin facts from overwhelming the frozen foundation MLLM.
The policy is task-neutral: it reads only evidence provenance, observation confidence, temporal
support counts, and explicit cross-fitted reliability bands. It never reads criterion labels,
ground truth, dataset identity, or a plugin task prediction.

## Evidence tiers

1. `calibrated_high`: a candidate passed a reliability policy fitted outside the target
   sample/fold. It remains advisory and never becomes ground truth.
2. `supported`: a positive observation has moderate/high raw confidence or at least three
   temporally supporting observations.
3. `context_only`: an unverified retrieval cue, low-confidence observation, deterministic
   summary, or fact without an explicit reliability signal.

All tiers require the frozen MLLM to inspect the real image and apply the Skill. Nondetection is
never converted into negative task evidence.

## Foundation branch arbitration

The full framework uses the same frozen foundation MLLM in three inference roles without updating
its parameters: (1) a Skill-only primary branch supplies the stable foundation prior; (2) a
Skill-plus-temporal branch supplies an alternate time-aware hypothesis; and (3) the final branch
receives both compact hypotheses, routed fact-only plugins, the Skill, and the same raw frames.
The final branch must inspect the images and resolve disagreement itself. Branch hypotheses are
neither small-model votes nor ground truth, and deterministic score averaging is prohibited.
To preserve local temporal context without restoring a long-video prompt, an auxiliary F candidate
reserves up to two neighboring sampled frames before lower-priority disagreements are selected.
This routing uses only Qwen branch states and frame chronology; it never reads task labels.

Core single-plugin ablations reuse the primary/temporal branches, so the full five-arm experiment
requires five Qwen judgments per sample rather than hiding extra fine-tuning or a small-model final
classifier. End-to-end full-framework latency nevertheless reports all three Qwen branch calls.

## Routing into the final decision prompt

- retain every `calibrated_high` and `supported` fact;
- retain at most a bounded, subject/predicate-diverse context tail when the plugin has at least
  one supported anchor;
- retain temporal retrieval cues when the adapter has already exported a bounded fact payload,
  because they only tell the MLLM when to inspect; they still cannot prove satisfaction;
- retain a bounded number of direct, inspectable observations tied to a real frame, timestamp,
  or box even when their raw confidence is not task-calibrated; prefer them over retrieval-only
  hypotheses, and require the MLLM to verify them in the real image;
- omit pure unverified retrieval candidates when they have neither a calibrated/supported anchor
  nor direct inspectable grounding;
- preserve an audit of original, retained, and prompt-omitted facts; never modify the upstream
  cache;
- plugins still cannot supply criterion probabilities, final labels, or verdicts.
