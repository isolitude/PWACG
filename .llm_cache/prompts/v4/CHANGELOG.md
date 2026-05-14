# v4 Changelog

## Anti-patterns added (from lasso regeneration failure analysis)
- DO NOT introduce merge_phi/propagate_f/propagate_phi helper abstractions
- DO NOT break dplex.dconstruct(const, theta) into sub-expressions
- DO NOT use self.BW for f-side propagators (use self.BW_fside)
- DO NOT pass tuples to vmapped functions without unpacking
- DO NOT refactor or "improve" the reference patterns

## Root cause
LLM tried to refactor the propagator dispatch pattern into generic helpers,
introducing tuple unpacking bugs and wrong physics expressions.
