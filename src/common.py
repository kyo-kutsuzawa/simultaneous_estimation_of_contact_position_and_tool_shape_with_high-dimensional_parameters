from dataclasses import dataclass, field
import enum
import typing
import numpy as np


class EstimationMethod(enum.Enum):
    unspecified = enum.auto()
    proposed = enum.auto()
    baseline = enum.auto()
    oracle = enum.auto()
    naive = enum.auto()


class ShapeType(enum.Enum):
    undefined = enum.auto()
    straight = enum.auto()
    arch = enum.auto()
    angular = enum.auto()
    wavy = enum.auto()
    knife = enum.auto()
    zigzag = enum.auto()
    discontinuous = enum.auto()

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.name < other.name
        return NotImplemented


@dataclass
class EstimationParams:
    pass


@dataclass
class ProposedParams(EstimationParams):
    n_particles: int = None
    """number of particles"""
    cell_size: int = None
    """Size of cells"""
    d_th: float = None
    """Hyper-parameter"""
    theta_th: float = None
    """Hyper-parameter"""
    s_inc: float = None
    """Hyper-parameter"""
    s_dec: float = None
    """Hyper-parameter"""
    std_x: np.ndarray = None
    """Standard deviation of state"""
    std_y: np.ndarray = None
    """Standard deviation of observation"""
    shape_extent: typing.Tuple[float, float, float, float] = None
    """Shape extent parameters"""
    eff_th: float = None
    """Threshold of the ratio of the effective number of particles"""


@dataclass
class BaselineParams(EstimationParams):
    rho: float = None
    """Forgetting factor of the recursive least squares"""


@dataclass
class OracleParams(EstimationParams):
    pass


@dataclass
class NaiveParams(EstimationParams):
    n_particles: int = None
    """number of particles"""
    cell_size: int = None
    """Size of cells"""
    std_x: np.ndarray = None
    """Standard deviation of state"""
    std_y: np.ndarray = None
    """Standard deviation of observation"""
    shape_extent: typing.Tuple[float, float, float, float] = None
    """Shape extent parameters"""
    eff_th: float = None
    """Threshold of the ratio of the effective number of particles"""


@dataclass
class EstimationResult:
    pos_true: list = field(default_factory=list)
    """Time series of true contact points"""
    force_true: list = field(default_factory=list)
    """Time series of true contact force/torque"""
    observations: list = field(default_factory=list)
    """Time series of observed contact force/torque"""
    pos_est: list = field(default_factory=list)
    """Time series of estimated contact positions"""
    shape_est: np.ndarray = None
    """Estimated shape parameters at the final time step"""
    shape_progress: list = field(default_factory=list)
    """Time series of estimated shape parameters"""
    shape_type: ShapeType = ShapeType.undefined
    """Shape type"""
    method: EstimationMethod = EstimationMethod.unspecified
    """Estimation method"""
    hyper_params: EstimationParams = None
    """Hyper-arameters for estimation"""
    force_fluctuation_amp: float = None
    """Amplitude of the contact force fluctuation; <0 for the default value"""
