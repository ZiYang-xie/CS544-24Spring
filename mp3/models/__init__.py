from .central_path import CentralPathSolver
from .primal_dual import PrimalDualSolver, PrimalDualSolver_NonSep
from .corrector_primal_dual import CorrectorPrimalDualSolver, CorrectorPrimalDualSolver_NonSep

MODEL_ZOO = {
    'centeral_path': CentralPathSolver,
    'primal_dual': PrimalDualSolver,
    # 'primal_dual_nosep': PrimalDualSolver_NonSep,
    'corrector_primal_dual': CorrectorPrimalDualSolver,
    # 'corrector_primal_dual_sep': CorrectorPrimalDualSolver_NonSep
}