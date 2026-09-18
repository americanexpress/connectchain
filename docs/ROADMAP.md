# ConnectChain Fork — Ideas & Future Work

> Maintained on the `wilsonhj/connectchain` fork. This lists forward-looking ideas, not commitments
> or dates — treat it as a backlog to pick from, not a schedule. For what's already been fixed, see
> `ARCHITECTURE.md`'s Bug History section.
> **Priority scale:** 🔴 High value · 🟡 Medium · 🟢 Nice-to-have / exploratory

---

## Summary

| ID | Type | Title | Priority |
|----|------|-------|----------|
| REF-1 | Refactor | Migrate `ValidLLMChain` off `LLMChain` onto a native LCEL `Runnable`, ahead of LangChain 1.0 removing `LLMChain` | 🔴 |
| F-1 | Feature | `MCPToolAgent` with LangGraph checkpointing (multi-turn memory) | 🟡 |
| F-2 | Feature | Async parallel tool dispatch via `asyncio.gather` | 🟡 |
| F-3 | Feature | `astream()` / token-streaming support in `PortableOrchestrator` | 🟡 |
| F-4 | Feature | Structured output enforcement (`with_structured_output`) | 🟡 |
| F-5 | Feature | Per-request retry + fallback model routing | 🟡 |
| F-6 | Feature | Prompt injection detection sanitizer (built-in) | 🟡 |
| F-7 | Feature | Agent team orchestration (supervisor + worker pattern) | 🟢 |
| OPT-1 | Optimization | `asyncio.Lock`-based session cache for async callers | 🟢 |
| OPT-2 | Optimization | Config hot-reload without restart | 🟢 |

Bug fixes are tracked as GitHub issues/PRs, not in this file — see the repo's issue tracker and
`ARCHITECTURE.md` §7 for what's already been found and fixed.

---

## Refactoring

### REF-1 — Migrate `ValidLLMChain` off `LLMChain`

**Scope:** `connectchain/chains/valid_llm_chain.py`, and anything constructing it directly.
**Why:** `ValidLLMChain` inherits `langchain.chains.llm.LLMChain`, which LangChain's own
`LangChainDeprecationWarning` says was deprecated in 0.1.0 and will be removed in LangChain 1.0.
`pyproject.toml` currently pins `langchain<0.4.0` to avoid that break; this refactor would be needed
before ever lifting that pin.

**Sketch (illustrative, not a finished design):**

```python
from langchain_core.runnables import RunnableLambda

class ValidChain:
    def __init__(self, llm, prompt, output_sanitizer=None):
        pipe = prompt | llm
        if output_sanitizer:
            pipe = pipe | RunnableLambda(output_sanitizer)
        self._chain = pipe

    def invoke(self, inputs: dict) -> str:
        return self._chain.invoke(inputs)

    async def ainvoke(self, inputs: dict) -> str:
        return await self._chain.ainvoke(inputs)
```

---

## New Features

### F-1 — MCPToolAgent + LangGraph Checkpointing 🟡

**Problem:** Current `MCPToolAgent` is stateless — no memory between turns, no interrupt/resume.
**Solution:** Integrate LangGraph `StateGraph` with `MemorySaver` checkpointer.

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

class StatefulMCPAgent:
    def __init__(self, tools, llm):
        self.graph = self._build_graph(tools, llm)
        self.checkpointer = MemorySaver()

    def run(self, query: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke({"messages": [query]}, config=config)
```

**Real-world use case:** Multi-turn financial Q&A agent that remembers account context across a session without re-fetching.

---

### F-2 — Parallel Tool Dispatch 🟡

**Problem:** Multi-tool agents call tools sequentially, creating latency proportional to tool count.
**Solution:** `asyncio.gather()` for independent tool calls.

```python
async def parallel_dispatch(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
    tasks = [self.call_tool(tc.name, tc.args) for tc in tool_calls]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**Expected speedup:** 3–5× for agents calling 3+ independent tools.

---

### F-3 — Streaming Response (`astream()`) 🟡

**Problem:** `PortableOrchestrator` supports sync (`run_sync()`) and async (`run()`) calls, but not
token-by-token streaming.
**Solution:** Expose `astream()`/`stream()` for streaming responses to the application layer.

```python
async def astream(self, input: str):
    async for chunk in self._chain.astream({"input": input}):
        yield chunk
```

---

### F-4 — Structured Output Enforcement 🟡

```python
from pydantic import BaseModel

class TransactionSummary(BaseModel):
    amount: float
    currency: str
    merchant: str
    risk_score: float

chain = prompt | llm.with_structured_output(TransactionSummary)
result: TransactionSummary = chain.invoke({"transaction": raw_data})
```

---

### F-5 — Retry + Fallback Model Routing 🟡

```python
from langchain_core.runnables import RunnableWithFallbacks

primary_chain = prompt | primary_llm
fallback_chain = prompt | fallback_llm

resilient_chain = primary_chain.with_fallbacks([fallback_chain])
```

---

### F-6 — Built-in Prompt Injection Detection 🟡

```python
INJECTION_PATTERNS = [
    r'ignore previous instructions',
    r'disregard (all|your) (prior|previous|system)',
    r'you are now (a |an )?(?!assistant)',
    r'<\s*script\s*>',
    r'system:\s*you are',
]

def detect_prompt_injection(query: str) -> str:
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            raise OperationNotPermittedException("Prompt injection detected")
    return query
```

---

### F-7 — Agent Team Orchestration (Supervisor + Worker) 🟢

```python
def supervisor_node(state):
    decision = supervisor_llm.invoke(state['messages'])
    return Command(goto=decision.next_worker)

graph.add_node("research_agent", research_subgraph)
graph.add_node("code_agent", code_subgraph)
graph.add_conditional_edges("supervisor", route_to_worker)
```

---

## Optimizations

### OPT-1 — Thread-Safe Session Token Cache 🟡

```python
import asyncio

class AsyncSessionMap:
    def __init__(self):
        self._sessions = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get_token(self, key: str) -> str:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        async with self._locks[key]:
            if self._is_expired(key):
                self._sessions[key] = await self._refresh_token(key)
            return self._sessions[key]['token']
```

### OPT-2 — Config Hot-Reload 🟢

```python
from watchfiles import awatch

async def watch_config(config_path: str, on_reload: callable):
    async for _ in awatch(config_path):
        on_reload(Config.from_file(config_path))
```
