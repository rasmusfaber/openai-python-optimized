# Modified by Rasmus Faber: point optional-dependency instructions to the fork distribution.
from .._exceptions import OpenAIError

INSTRUCTIONS = """

OpenAI error:

    missing `{library}`

This feature requires additional dependencies:

    $ pip install openai-python-optimized[{extra}]

"""


def format_instructions(*, library: str, extra: str) -> str:
    return INSTRUCTIONS.format(library=library, extra=extra)


class MissingDependencyError(OpenAIError):
    pass
