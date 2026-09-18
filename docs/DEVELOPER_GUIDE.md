# ConnectChain Fork — Developer Guide

> **Audience:** Engineers contributing to `wilsonhj/connectchain`
> **Prerequisites:** Python 3.11+, `uv`, basic LangChain familiarity

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Configuration Schema](#2-configuration-schema)
3. [Adding a Sanitizer](#3-adding-a-sanitizer)
4. [Writing Tests](#4-writing-tests)
5. [Debugging Tips](#5-debugging-tips)
6. [Code Style & Linting](#6-code-style--linting)
7. [Branch & PR Conventions](#7-branch--pr-conventions)

---

## 1. Environment Setup

```bash
# 1. Clone your fork
git clone https://github.com/wilsonhj/connectchain.git
cd connectchain

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies including dev extras
uv sync --dev

# 4. Copy and configure environment files
cp example.env .env
cp connectchain/config/example.config.yml config.yml
# Edit .env and config.yml with your API keys / EAS credentials

# 5. Run the test suite to verify setup
make test
# All tests should pass
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CONFIG_PATH` | Yes | Absolute path to your `config.yml` (read by `connectchain.utils.Config.from_env()`) |
| `OPENAI_API_KEY` | For direct access | OpenAI API key |
| *(name set by `eas.id_key`/`eas.secret_key` in config.yml)* | For EAS auth | EAS credential env vars are **not** fixed names — `config.yml`'s `eas.id_key`/`eas.secret_key` point to whichever env vars you use (see `example.env`'s `CONSUMER_ID1`/`CONSUMER_SECRET1`) |
| `REQUESTS_CA_BUNDLE` | Optional | Path to a `.crt` CA bundle for outbound TLS, see `example.env`. Outbound proxying is **not** an env var — configure it via the `proxy: {host, port}` block in `config.yml` (see the schema below) |

---

## 2. Configuration Schema

```yaml
# config.yml
models:
  '1':                          # Model index (string key)
    provider: openai            # openai | azure | anthropic
    type: chat                  # chat | completion
    model_name: gpt-4o-mini
    bypass_eas: false           # true = direct API, skip EAS
    # Optional overrides:
    eas:
      id_key: MY_EAS_ID
      secret_key: MY_EAS_SECRET
      scope: ["api://your-scope"]
    proxy:
      host: proxy.corp.com
      port: 8080
    cert:
      cert_path: /etc/ssl/certs
      cert_name: corp.crt
      cert_size: 2048           # optional; omitted/falsy skips the downloaded-file size check
```

### Direct Access (No EAS)

Omit `eas`, `proxy`, and `cert` blocks entirely, or set `bypass_eas: true` per model.

---

## 3. Adding a Sanitizer

Sanitizers are plain Python callables: `(str) -> str`. They raise `OperationNotPermittedException` to block execution.

### Input Sanitizer (ValidPromptTemplate)

```python
from connectchain.prompts import ValidPromptTemplate
from connectchain.utils.exceptions import OperationNotPermittedException
import re

def block_pii(query: str) -> str:
    """Block prompts containing SSN patterns."""
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', query):
        raise OperationNotPermittedException(f"PII detected in prompt: {query[:50]}...")
    return query

prompt = ValidPromptTemplate(
    input_variables=["question"],
    template="Answer this: {question}",
    output_sanitizer=block_pii,
)
```

Note the `ValidPromptTemplate` constructor arg is named `output_sanitizer` too, but it validates
the *rendered prompt* before it's sent to the LLM — separate from `ValidLLMChain.output_sanitizer`
below, which validates the LLM's *response*.

### Output Sanitizer

```python
from connectchain.chains import ValidLLMChain

def redact_output(response: str) -> str:
    """Redact credit card numbers from LLM output."""
    return re.sub(r'\b(?:\d[ -]?){13,16}\b', '[REDACTED]', response)

chain = ValidLLMChain(llm=llm, prompt=prompt, output_sanitizer=redact_output)
```

`ValidLLMChain.output_sanitizer` is applied to the LLM's response on all four dispatch paths
(`run()`, `arun()`, `invoke()`, `ainvoke()`), not to the input. If you build a `PortableOrchestrator`
via `from_prompt_template(...)`, pass `output_sanitizer=` as a kwarg there to have it forwarded to
the underlying `ValidLLMChain`.

---

## 4. Writing Tests

Tests live in `tests/unit_tests/` and `tests/integration_tests/` and use `unittest.TestCase` (not
bare pytest-style classes), matching the rest of the suite:

```python
# tests/unit_tests/test_my_sanitizer.py
# (include the standard Apache 2.0 license header at the top of any new file --
#  copy it from an existing test file)
import unittest

from connectchain.utils.exceptions import OperationNotPermittedException
from mymodule import block_pii


class TestBlockPII(unittest.TestCase):
    def test_clean_input_passes(self):
        result = block_pii("What is the weather?")
        self.assertEqual(result, "What is the weather?")

    def test_ssn_raises(self):
        with self.assertRaises(OperationNotPermittedException):
            block_pii("My SSN is 123-45-6789")

    def test_partial_ssn_passes(self):
        result = block_pii("Call 555-1234")
        self.assertEqual(result, "Call 555-1234")
```

```bash
# Run just your new test
uv run pytest tests/unit_tests/test_my_sanitizer.py -v

# Run with coverage
make test-unit-cov
```

---

## 5. Debugging Tips

### SessionMap first-lookup behavior

`SessionMap` is a per-process singleton. `is_expired(session_id)` and `get_valid_llm(session_id)`
both return safely (`True`/`None`) for a `session_id` that was never registered — they do not raise
`KeyError`. `get_llm(session_id)` is the one exception: it does a raw dict lookup and raises
`KeyError` if the session isn't registered, so only call it after confirming `is_expired()` is
`False` (or use `get_valid_llm()`, which does both atomically).

### `LangChainDeprecationWarning` if you see it

If you're on a version of this codebase or a dependency that still calls a deprecated LangChain API
directly (`Chain.run()`/`Chain.arun()`), you'll see:

```
LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0
and will be removed in 1.0. Use :meth:`~invoke` instead.
```

`PortableOrchestrator` already uses `.invoke()`/`.ainvoke()`, so you shouldn't see this from
ConnectChain's own code paths on current `main` — if you do, it's worth filing an issue.

### Verbose LangChain Logging

```python
import langchain
langchain.debug = True  # Prints full chain inputs/outputs
```

### Config Not Loading

```bash
# Check env var is set
echo $CONFIG_PATH
# Should print absolute path to config.yml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yml'))"
```

---

## 6. Code Style & Linting

```bash
# Lint (pylint, configured via .pylintrc)
make lint

# Type checking
uv run mypy connectchain/

# Format (black)
uv run black connectchain/ tests/
```

- Max line length: **100** characters, enforced by black/isort (see `pyproject.toml`); pylint's own limit is 160 (see `.pylintrc`)
- Docstrings: Google style
- Type hints: required for all public functions

---

## 7. Branch & PR Conventions

| Type | Branch Pattern | Example |
|------|---------------|------|
| Bug fix | `fix/<issue-id>-short-description` | `fix/bug-1-session-map-keyerror` |
| Feature | `feat/<ticket>-short-description` | `feat/f1-mcp-langgraph-memory` |
| Docs | `docs/<description>` | `docs/connectchain-architecture-guide` |
| Refactor | `refactor/<scope>` | `refactor/lcel-migration` |
| Chore | `chore/<scope>` | `chore/bump-langchain-0.3` |

### Commit Message Format

```
<type>(<scope>): <short summary>

<body: what changed and why>

Fixes #<issue-number>
```

PR titles must match the commit convention. Fill out `.github/pull_request_template.md` completely before requesting review.
