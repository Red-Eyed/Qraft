# Rules

`Rules[Algorithm]` resolves a node against a default and ordered `Rule` entries.
The last matching rule wins. `Exclude` retains the caller's reason at resolution.
The final plan lists excluded identities.

`ByOperator`, `ByName`, and `ByTensor` implement the small `Selector` protocol.
Custom selectors receive a node and domain graph, so topology predicates need no
backend object. Resolution is pure. Callers assemble implementations at the edge;
there is no global registry or import-time registration.
