from textwrap import dedent
from typing import Tuple


_AVAILABLE_TOOLS_BLOCK = """
TOOLS AVAILABLE
---------------
1. arxiv_search(query: str, limit: int) – returns list of papers
"""

_SYSTEM_TEMPLATE = dedent(
    """\
    You are an **orchestration planner**.
    
    Your job:
      1. Read the USER_TASK.
      2. Decide which worker roles and tools are required.
      3. Produce a JSON plan that exactly matches this schema:
    
    ```json
    {{
      "steps": [
        {{
          "id": "s1",
          "action": "spawn" | "tool" | "merge",
          "worker": "ResearchWorker" | "CodeWorker" | "WriterWorker" | null,
          "tool":   "arxiv_search" | null,
          "args":   {{ ... }} | null,
          "depends_on": ["s0", "s3", ...]   // IDs of earlier steps
        }}
      ]
    }}
    ```

    • Use `"spawn"` to start a worker of the given role.
    • Use `"tool"`  to call a stateless tool.
    • The final step **must** contain the full answer in its output.

    Return **only** valid JSON – no markdown, no commentary.
    """
)

_USER_TEMPLATE = dedent(
    """\
    USER_TASK:
    ----------
    {task}

    (Respond with the JSON plan.)"""
)


def build_planner_prompts(task: str) -> Tuple[str, str]:
    """
    Parameters
    ----------
    task : str
        Natural-language request from the end-user.

    Returns
    -------
    system_prompt : str
        Fixed instructions that define the planner’s role and JSON schema.
    user_prompt : str
        The task embedded in a short wrapper that triggers the reply.
    """
    return _SYSTEM_TEMPLATE + _AVAILABLE_TOOLS_BLOCK, _USER_TEMPLATE.format(task=task)
