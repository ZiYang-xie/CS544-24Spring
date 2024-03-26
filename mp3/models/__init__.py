from .central_path import CentralPathSolver
from .primal_dual import PrimalDualSolver
from .corrector_primal_dual import CorrectorPrimalDualSolver

MODEL_ZOO = {
    # 'centeral_path': CentralPathSolver,
    'primal_dual': PrimalDualSolver,
    'corrector_primal_dual': CorrectorPrimalDualSolver
}