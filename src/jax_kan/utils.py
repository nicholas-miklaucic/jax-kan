import io
import re
from abc import ABCMeta
from dataclasses import asdict, is_dataclass
from functools import partial
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import flax.linen as nn
import humanize
import jax
import jax.numpy as jnp
import numpy as np
import rich
from flax.serialization import msgpack_restore, msgpack_serialize
from jaxtyping import jaxtyped
from rich.style import Style
from rich.text import Text
from rich.tree import Tree


class Identity(nn.Module):
    """Identity function, useful for generic modules."""

    @nn.compact
    def __call__(self, x):
        return x


INFERNA = [
    '#e6ab00',
    '#eb9900',
    '#ec8800',
    '#e97800',
    '#e46a25',
    '#db5e40',
    '#cf5352',
    '#c14b60',
    '#b5416e',
    '#a5387c',
    '#93318a',
    '#7f2c95',
    '#67289e',
    '#4d25a3',
    '#2f23a3',
    '#00229c',
]

from rich.console import Console
from rich.highlighter import RegexHighlighter
from rich.theme import Theme


def format_scalar(scalar: int | float, chars=6) -> str:
    """Formats the scalar using no more than the given number of characters, if possible."""
    if scalar < 0:
        return '-' + format_scalar(scalar, chars=chars - 1)
    if isinstance(scalar, int):
        if len(str(scalar)) - chars > 3:
            # egregiously longer, use scientific notation
            used = len(f'{scalar:.0E}')
            remaining = max(chars - used, 0)
            return '{:.dE}'.replace('d', str(remaining)).format(scalar)
        else:
            return str(scalar)
    else:
        if scalar == int(scalar):
            return format_scalar(int(scalar), chars=chars - 1) + '.'

        decimal = str(scalar)
        if decimal.startswith('0.'):
            frac = decimal.removeprefix('0.')
            wasted = len(frac.lstrip('0')) - len(decimal.replace('.', ''))
        else:
            wasted = len(decimal.replace('.', '').rstrip('0')) - len(decimal.replace('.', ''))

        suffix = f'{scalar:.0e}'[1:]
        if wasted > len(suffix):
            used = len(f'{scalar:.0E}')
            remaining = max(chars - used, 0)
            return '{:.dE}'.replace('d', str(remaining)).format(scalar)
        else:
            remaining = max(chars - 2, 0)
            return '{:.df}'.replace('d', str(remaining)).format(scalar)


def item_if_arr(x: int | float | jax.Array) -> float:
    if isinstance(x, (int, float)):
        return x
    else:
        return x.item()


def load_pytree(file: PathLike):
    """Loads a MsgPack serialized PyTree."""
    with open(Path(file), 'rb') as infile:
        return msgpack_restore(infile.read())


def save_pytree(obj, file: PathLike):
    """Saves a MsgPack serialized PyTree."""
    with open(Path(file), 'wb') as out:
        out.write(msgpack_serialize(obj))


class TreeVisitor:
    def __init__(self):
        pass

    def jax_arr(self, arr: jax.Array):
        raise NotImplementedError()

    def scalar(self, x: int | float):
        raise NotImplementedError()

    def np_arr(self, arr: np.ndarray):
        raise NotImplementedError()


class StatVisitor(TreeVisitor):
    def __init__(self) -> None:
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        flat = arr.flatten().astype(jnp.float32)
        inds = 1 + 0.01 * jnp.cos(jnp.arange(len(flat), dtype=jnp.float32))
        return f'{(flat * inds).mean().item():.4f}'

    def scalar(self, x: int | float):
        return f'{x:.4f}' + x.__class__.__name__[0]

    def np_arr(self, arr: np.ndarray):
        return self.jax_arr(jnp.array(arr))


class StructureVisitor(TreeVisitor):
    def __init__(self):
        super().__init__()

    def jax_arr(self, arr: jax.Array):
        return f'{arr.dtype}{list(arr.shape)}'

    def scalar(self, x: int | float):
        return f'{x.__class__.__name__}=' + str(x)

    def np_arr(self, arr: np.ndarray):
        return f'np{list(arr.shape)}'


COLORS = [
    '#00a0ec',
    '#00bc70',
    '#deca00',
    '#ff7300',
    '#d83990',
    '#7555d3',
    '#8ac7ff',
    '#00f0ff',
    '#387200',
    '#aa2e00',
    '#ff7dc6',
    '#e960ff',
]

KWARGS = [dict(color=color) for color in COLORS]
KWARGS[0]['bold'] = True

STYLES = [Style(**kwargs) for kwargs in KWARGS]


def tree_from_dict(base: Tree, obj, depth=0, collapse_single=True):
    style = STYLES[depth % len(STYLES)]
    if isinstance(obj, dict):
        if len(obj) == 1:
            k, v = next(iter(obj.items()))
            base.label = base.label + ' >>> ' + k
            tree_from_dict(base, v, depth)
        else:
            for k, v in obj.items():
                child = base.add(k, style=style)
                tree_from_dict(child, v, depth + 1)
    else:
        base.add(obj, style=style)


def tree_traverse(visitor: TreeVisitor, obj, max_depth=2, collapse_single=True):
    if isinstance(obj, jax.Array):
        return visitor.jax_arr(obj)
    elif isinstance(obj, np.ndarray):
        return visitor.np_arr(obj)
    elif isinstance(obj, (list, tuple)):
        if max_depth == 0:
            return '[...]'
        else:
            if collapse_single and len(obj) == 1:
                new_depth = max_depth
            else:
                new_depth = max_depth - 1

            return {str(i): tree_traverse(visitor, child, new_depth) for i, child in enumerate(obj)}
    elif isinstance(obj, (float, int)):
        return visitor.scalar(obj)
    elif isinstance(obj, dict):
        if max_depth == 0:
            return '{...}'
        else:
            if collapse_single and len(obj) == 1:
                new_depth = max_depth
            else:
                new_depth = max_depth - 1

            excluded = (('parent', None), ('name', None))
            return {k: tree_traverse(visitor, v, new_depth) for k, v in obj.items() if (k, v) not in excluded}
    elif is_dataclass(obj):
        return {obj.__class__.__name__: tree_traverse(visitor, asdict(obj), max_depth)}
    else:
        name = getattr(obj, '__name__', '|')
        return f'{obj.__class__.__name__}={name}'


def show_obj(obj):
    # print_json(data=obj)
    for k, v in obj.items():
        tree = Tree(label=k, style=STYLES[0])
        tree_from_dict(tree, v)
        rich.print(tree)


def _debug_structure(tree_depth=5, **kwargs):
    """Prints out the structure of the inputs."""
    show_obj({f'{k}': tree_traverse(StructureVisitor(), v, tree_depth) for k, v in kwargs.items()})


def _debug_stat(tree_depth=5, **kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries."""
    show_obj({f'{k}': tree_traverse(StatVisitor(), v, tree_depth) for k, v in kwargs.items()})


def debug_structure(*args, **kwargs):
    """Prints out the structure of the inputs. Returns first argument."""
    new_kwargs = {f'arg{i}': arg for i, arg in enumerate(args)}
    new_kwargs.update(**kwargs)
    jax.debug.callback(_debug_structure, **new_kwargs)
    return list(new_kwargs.values())[0]


def debug_stat(*args, **kwargs):
    """Prints out a reduction of the inputs. Is almost the mean, but with a small fudge factor so differently-shaped arrays will have different summaries. Returns first argument."""
    new_kwargs = {f'arg{i}': arg for i, arg in enumerate(args)}
    new_kwargs.update(**kwargs)
    jax.debug.callback(_debug_stat, **new_kwargs)
    return list(new_kwargs.values())[0]


def flax_summary(
    mod: nn.Module,
    rngs={},
    *args,
    compute_flops=True,
    compute_vjp_flops=True,
    console_kwargs=None,
    table_kwargs=MappingProxyType({'safe_box': False, 'expand': True, 'box': rich.box.SIMPLE}),
    column_kwargs=MappingProxyType({'justify': 'right'}),
    show_repeated=False,
    depth=None,
    colorize=True,
    **kwargs,
):
    out = mod.tabulate(
        dict(params=jax.random.key(0), **rngs),
        *args,
        compute_flops=compute_flops,
        compute_vjp_flops=compute_vjp_flops,
        console_kwargs=console_kwargs,
        table_kwargs=table_kwargs,
        column_kwargs=column_kwargs,
        show_repeated=show_repeated,
        depth=depth,
        **kwargs,
    )

    # hack to control numbers so they're formatted reasonably
    # 12580739072 flops is not very helpful

    max_color_i = len(INFERNA) - 1

    nums = re.findall(r'\d' * 4 + '+', out)
    color_max = max(len(n) for n in nums) + 1

    def _colorize(m: re.Match):
        s = m.group(0)

        n = int(s.strip())
        e = np.log10(np.abs(n) + 1)

        color_i = int(round(max_color_i * np.clip(e / color_max, 0, 1)))

        color = INFERNA[color_i]
        t = Text(s.strip(), style=Style(color=color, bold=(e > 3)))
        f = io.StringIO()
        console = rich.console.Console(
            file=f, color_system=rich.console.Console().color_system, **(console_kwargs or {})
        )
        console.print(t, end='')
        return f.getvalue()

    def human_units(m: re.Match):
        """Format using units, preserving the length with spaces."""
        human = humanize.metric(int(m.group(0))).replace(' ', '')
        pad_num = len(m.group(0)) - len(human)
        return ' ' * pad_num + human

    if colorize:
        out = re.sub(r'\d' * 4 + '+', _colorize, out)
    out = re.sub(r'\d' * 6 + '+', human_units, out)

    print(out)
    return out
