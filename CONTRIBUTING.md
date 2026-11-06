# Contribution and style guide

Keep changes small, modular, and consistent with the package boundaries in
[`docs/architecture.md`](docs/architecture.md).

## Python formatting

- Use UTF-8, LF line endings, spaces instead of tabs, and four spaces per
  indentation level.
- Keep source lines at or below 85 characters. Wrap long calls and expressions
  with parentheses; do not use backslash continuations.
- Use double-quoted strings, trailing commas in multiline literals and calls,
  and two blank lines between top-level definitions.
- Let Ruff format Python code and check lint rules:

  ```bash
  ruff format .
  ruff check .
  ```

- Keep imports grouped as standard library, third-party packages, then local
  package imports. Ruff sorts these groups.
- Use `snake_case` for modules, functions, variables, and parameters;
  `PascalCase` for classes and exceptions; and `UPPER_SNAKE_CASE` for constants.
- Add type annotations to public functions and methods. Avoid `Any` where a
  useful protocol, type variable, or concrete type can express the contract.
- Keep functions focused. Prefer explicit configuration and dependency passing
  over hidden global state or simulator imports at module import time.

## Comments and docstrings

- Write comments and docstrings in English, in complete sentences, with a
  period at the end. User-facing guides may be bilingual when useful.
- Use `""" triple double quotes """` for module, class, and public function
  docstrings.
- Public APIs should document purpose, arguments, return values, and raised
  exceptions when these are not obvious from the signature. Use Google-style
  sections (`Args:`, `Returns:`, `Raises:`) for multi-line API documentation.
- Comments should explain intent, constraints, or non-obvious reasons. Avoid
  comments that merely restate the next line of code.
- Mark follow-up work as `TODO(owner): ...` or `TODO(issue-id): ...`; include a
  concrete action rather than an open-ended note.
- Keep implementation detail out of public docstrings unless callers need it
  to use the API correctly. Put architecture rationale in `docs/`.

## Configuration and documentation

- Use four spaces for Python and TOML indentation. YAML uses its conventional
  two-space indentation; never use tabs.
- Keep configuration declarative and document non-obvious units, coordinate
  frames, update rates, and defaults next to the setting or in its schema.
- Use relative links within the repository and fenced code blocks with a
  language identifier in Markdown.

## Before submitting

Run formatting and lint checks for changed Python files:

```bash
ruff format .
ruff check .
```

Run relevant tests when changing behavior. Keep simulator-dependent checks
separate from checks that can run without Isaac Lab or mjlab installed.
