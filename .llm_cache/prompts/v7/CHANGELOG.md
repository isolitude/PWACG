# v6 Changelog

## Strengthened Anti-pattern 6: Sbc-first argument order

Made the anti-pattern much more explicit:
- Concrete function signatures with exact parameter names (not placeholders)
- Correct and incorrect callsite examples
- Listed all 6 affected methods by name
- Added as "MOST COMMON bug" header

This was necessary because v5's abstract description was not strong enough to
prevent the LLM from generating amp-first signatures.

Also updated artifact_registry.py task descriptions for pull and lasso to
include explicit Sbc-first instructions.
