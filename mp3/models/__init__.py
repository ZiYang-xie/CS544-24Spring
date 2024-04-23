from .central_path import CentralPathSolver
from .primal_dual import PrimalDualSolver, PrimalDualSolver_NonSep
from .corrector_primal_dual import CorrectorPrimalDualSolver, CorrectorPrimalDualSolver_NonSep


def build_cp_variants(update_t):
    class CentralPathSolver_t(CentralPathSolver):
        def __init__(self, c, A, b, t=1, update_t=update_t, use_wandb=False, vis=True):
            super().__init__(c, A, b, t, update_t, use_wandb, vis)
    return CentralPathSolver_t

MODEL_ZOO = {
    'centeral_path': CentralPathSolver,
    # **{f'centeral_path_{t}': build_cp_variants(t) for t in [1.1, 1.2, 1.3, 1.4, 1.5]},
    'primal_dual': PrimalDualSolver,
    # 'primal_dual_nosep': PrimalDualSolver_NonSep,
    # 'corrector_primal_dual': CorrectorPrimalDualSolver,
    # 'corrector_primal_dual_nosep': CorrectorPrimalDualSolver_NonSep
}
