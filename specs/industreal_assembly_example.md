# Industrial-like assembly assessment

Assess whether the operator correctly completes the base assembly, then the
wheel assembly, and finally verifies the final assembly. Each completed state
must persist. The procedure fails when a required state is reached in the wrong
order, a step is omitted, an incorrect part is used, or the final state is not
visibly completed. Return uncertain when the relevant state is not observable.
