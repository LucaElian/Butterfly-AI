from .intent_routing import build as build_intent_routing
from .comprehension import build as build_comprehension
from .binding import build as build_binding
from .arithmetic import build as build_arithmetic
from .epistemic import build as build_epistemic
from .instruction_following import build as build_instruction_following

BUILDERS = {
    "intent_routing": build_intent_routing,
    "comprehension": build_comprehension,
    "binding": build_binding,
    "arithmetic": build_arithmetic,
    "epistemic": build_epistemic,
    "instruction_following": build_instruction_following,
}
