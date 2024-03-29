import scipy.optimize as opt

def minimize_with_restart(function, initial_guess, method='CG', jac=None, tol=1e-20,
                      options={
                          'gtol': 1e-2,
                          'disp': True,
                          'maxiter': 50,
                          'return_all': True}):
    return_data = []
    iter = 0
    while iter < 10:
        result = opt.minimize(function, initial_guess, method=method, jac=jac, tol=tol, options=options)
        return_data.append(result)
        if result.success:
            break
        # print(f"result.x:{result.x.shape}")
        initial_guess = result.x
        print(f"iter: {iter} success: {result.success}")
        iter += 1
    final_results = return_data[-1]
    all_vects = []
    for res in return_data:
        all_vects.extend(res.allvecs)
    final_results.allvecs = all_vects
    return final_results