import abc
from typing import Optional, TypeVar, Union

from equinox.internal import AbstractVar
from jaxtyping import Array, PyTree

from .._custom_types import (
    AbstractBrownianIncrement,
    BrownianIncrement,
    RealScalarLike,
    SpaceTimeLevyArea,
    Y,
    Args
)
from .._path import AbstractPath
from .._term import AbstractTerm


_Control = TypeVar("_Control", bound=Union[PyTree[Array], AbstractBrownianIncrement])
_PathState = TypeVar("_PathState")

class AbstractBrownianPath(AbstractPath[_Control]):
    """Abstract base class for all Brownian paths."""

    levy_area: AbstractVar[type[Union[BrownianIncrement, SpaceTimeLevyArea]]]

    @abc.abstractmethod
    def init(
        self,
        terms: PyTree[AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> _PathState:
        """Initialises any hidden state for the path.

        **Arguments** as [`diffrax.diffeqsolve`][].

        **Returns:**

        The initial path state, which should be used the first time `evaluate` is called.
        """

    @abc.abstractmethod
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = True,
        use_levy: bool = False,
        path_state: Optional[_PathState] = None,
    ) -> _Control:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the path at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Ignored. (This determines whether to treat the path as
            left-continuous or right-continuous at any jump points, but Brownian
            motion has no jump points.)
        - `use_levy`: If True, the return type will be a `LevyVal`, which contains
            PyTrees of Brownian increments and their Lévy areas.
        - `path_state`: If passed, the current state of the path.

        **Returns:**

        If `t1` is not passed:

        The value of the Brownian motion at `t0`.

        If `t1` is passed:

        The increment of the Brownian motion between `t0` and `t1`.
        """
