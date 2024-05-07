from beartype import beartype
from jaxtyping import jaxtyped

tcheck = jaxtyped(typechecker=beartype)
class_tcheck = beartype
