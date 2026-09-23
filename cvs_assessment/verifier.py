from __future__ import annotations

from .schema import (
    AssessmentPlan, CriterionEvidence, CriterionVerdict, EvidenceInterval,
    VerificationExpression,
)


class ExplicitVerifier:
    def __init__(
        self, pass_confidence: float, uncertain_confidence: float,
        fail_confidence: float | None = None, min_assessable_coverage: float = 0.5,
        min_negative_points: int = 2,
    ) -> None:
        self.pass_confidence = pass_confidence
        self.uncertain_confidence = uncertain_confidence
        self.fail_confidence = fail_confidence if fail_confidence is not None else pass_confidence
        self.min_assessable_coverage = min_assessable_coverage
        self.min_negative_points = min_negative_points

    def verify(
        self, plan: AssessmentPlan,
        evidence: dict[str, CriterionEvidence | list[EvidenceInterval]],
    ) -> tuple[list[CriterionVerdict], str, float]:
        if plan.invalid_specification:
            return [], "uncertain", 0.0
        results: list[CriterionVerdict] = []
        for criterion in plan.criteria:
            structured = evidence.get(criterion.key, CriterionEvidence())
            if isinstance(structured, list):
                # Compatibility for callers using the original positive-only API.
                structured = CriterionEvidence(positive_intervals=structured)
            intervals = structured.positive_intervals
            best = max(intervals, key=lambda item: item.confidence, default=None)
            if best is None:
                explicit_failure = (
                    bool(criterion.negative_evidence)
                    and structured.negative_point_count >= self.min_negative_points
                    and structured.assessable_coverage >= self.min_assessable_coverage
                    and structured.explicit_negative_confidence >= self.fail_confidence
                )
                if explicit_failure:
                    results.append(CriterionVerdict(
                        key=criterion.key, verdict="fail",
                        confidence=structured.explicit_negative_confidence,
                        assessable_coverage=structured.assessable_coverage,
                        positive_point_count=structured.positive_point_count,
                        negative_point_count=structured.negative_point_count,
                        unknown_point_count=structured.unknown_point_count,
                        reason=(
                            "Explicit negative evidence was observed with sufficient assessable "
                            f"coverage ({structured.assessable_coverage:.2f}). Rule: {criterion.decision_rule}"
                        ),
                    ))
                else:
                    results.append(CriterionVerdict(
                        key=criterion.key, verdict="uncertain",
                        confidence=0.0,
                        assessable_coverage=structured.assessable_coverage,
                        positive_point_count=structured.positive_point_count,
                        negative_point_count=structured.negative_point_count,
                        unknown_point_count=structured.unknown_point_count,
                        reason=(
                            "No sustained positive interval was found, but explicit negative evidence "
                            "or assessable visual coverage was insufficient for failure."
                        ),
                    ))
            elif best.confidence >= self.pass_confidence:
                results.append(CriterionVerdict(
                    key=criterion.key, verdict="pass", confidence=best.confidence,
                    evidence_intervals=intervals,
                    assessable_coverage=structured.assessable_coverage,
                    positive_point_count=structured.positive_point_count,
                    negative_point_count=structured.negative_point_count,
                    unknown_point_count=structured.unknown_point_count,
                ))
            elif best.confidence >= self.uncertain_confidence:
                results.append(CriterionVerdict(
                    key=criterion.key, verdict="uncertain", confidence=best.confidence,
                    evidence_intervals=intervals,
                    assessable_coverage=structured.assessable_coverage,
                    positive_point_count=structured.positive_point_count,
                    negative_point_count=structured.negative_point_count,
                    unknown_point_count=structured.unknown_point_count,
                    reason="Evidence is sustained but does not reach the required confidence.",
                ))
            else:
                results.append(CriterionVerdict(
                    key=criterion.key, verdict="uncertain", confidence=best.confidence,
                    evidence_intervals=intervals,
                    assessable_coverage=structured.assessable_coverage,
                    positive_point_count=structured.positive_point_count,
                    negative_point_count=structured.negative_point_count,
                    unknown_point_count=structured.unknown_point_count,
                    reason="Positive evidence exists but is below the verification threshold; it is not explicit negative evidence.",
                ))

        if plan.verification_expression is not None:
            overall, confidence = self._evaluate_expression(
                plan.verification_expression,
                {item.key: item for item in results},
                {
                    key: value if isinstance(value, CriterionEvidence)
                    else CriterionEvidence(positive_intervals=value)
                    for key, value in evidence.items()
                },
            )
            return results, overall, confidence

        verdicts = [item.verdict for item in results]
        if plan.logic == "AND":
            # For conjunction, one explicit contradiction is sufficient to
            # reject the complete procedure even if another criterion is unknown.
            overall = "fail" if "fail" in verdicts else "uncertain" if "uncertain" in verdicts else "pass"
        else:
            overall = "pass" if "pass" in verdicts else "uncertain" if "uncertain" in verdicts else "fail"
        confidence = min((item.confidence for item in results), default=0.0) if plan.logic == "AND" else max((item.confidence for item in results), default=0.0)
        return results, overall, confidence

    def _evaluate_expression(
        self,
        expression: VerificationExpression,
        results: dict[str, CriterionVerdict],
        evidence: dict[str, CriterionEvidence],
    ) -> tuple[str, float]:
        operator = expression.operator.upper()
        if operator == "CRITERION":
            result = results.get(expression.criterion_id or "")
            if result is None:
                return "uncertain", 0.0
            if expression.expected_verdict == "pass":
                return result.verdict, result.confidence
            matched = result.verdict == expression.expected_verdict
            return ("pass" if matched else "uncertain"), result.confidence

        if operator in {"ALL", "AND", "ANY", "OR"}:
            children = [
                self._evaluate_expression(child, results, evidence)
                for child in expression.arguments
            ]
            if not children:
                return "uncertain", 0.0
            verdicts = [verdict for verdict, _ in children]
            confidences = [confidence for _, confidence in children]
            if operator in {"ALL", "AND"}:
                verdict = (
                    "fail" if "fail" in verdicts else
                    "uncertain" if "uncertain" in verdicts else "pass"
                )
                return verdict, min(confidences)
            verdict = (
                "pass" if "pass" in verdicts else
                "uncertain" if "uncertain" in verdicts else "fail"
            )
            return verdict, max(confidences)

        if operator in {"BEFORE", "AFTER"}:
            left = evidence.get(expression.left_criterion or "", CriterionEvidence())
            right = evidence.get(expression.right_criterion or "", CriterionEvidence())
            if not left.positive_intervals or not right.positive_intervals:
                return "uncertain", 0.0
            left_time = min(interval.start_s for interval in left.positive_intervals)
            right_time = min(interval.start_s for interval in right.positive_intervals)
            satisfied = left_time < right_time if operator == "BEFORE" else left_time > right_time
            confidence = min(
                max(interval.confidence for interval in left.positive_intervals),
                max(interval.confidence for interval in right.positive_intervals),
            )
            return ("pass" if satisfied else "fail"), confidence

        if operator == "NOT" and len(expression.arguments) == 1:
            verdict, confidence = self._evaluate_expression(
                expression.arguments[0], results, evidence,
            )
            inverted = {"pass": "fail", "fail": "pass", "uncertain": "uncertain"}[verdict]
            return inverted, confidence
        raise ValueError(f"Unsupported verification operator: {expression.operator}")
