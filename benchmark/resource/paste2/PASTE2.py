import numpy as np
import ot
from scipy.spatial import distance

from helper import kl_divergence, intersect, to_dense_array, extract_data_matrix, generalized_kl_divergence, \
    high_umi_gene_distance, pca_distance, glmpca_distance


def gwloss_partial(C1, C2, T, loss_fun='square_loss'):
    g = gwgrad_partial(C1, C2, T, loss_fun) * 0.5
    return np.sum(g * T)


def wloss(M, T):
    return np.sum(M * T)


def fgwloss_partial(alpha, M, C1, C2, T, loss_fun='square_loss'):
    return (1 - alpha) * wloss(M, T) + alpha * gwloss_partial(C1, C2, T, loss_fun)


def print_fgwloss_partial(alpha, M, C1, C2, T, loss_fun='square_loss'):
    print("W term is: " + str((1 - alpha) * wloss(M, T)))
    print("GW term is: " + str(alpha * gwloss_partial(C1, C2, T, loss_fun)))


def gwgrad_partial(C1, C2, T, loss_fun="square_loss"):
    """Compute the GW gradient, as one term in the FGW gradient.

    Note: we can not use the trick in Peyre16 as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source cost matrix

    C2: array of shape (n_q,n_q)
        intra-target cost matrix

    T : array of shape(n_p, n_q)
        Transport matrix

    loss_fun

    Returns
    -------
    numpy.array of shape (n_p, n_q)
        gradient
    """
    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    #cC1 = np.dot(C1 ** 2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    A = np.dot(
        f1(C1),
        np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1))
    )

    #cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2 ** 2 / 2)
    B = np.dot(
        np.dot(np.ones(C1.shape[0]).reshape(1, -1), T),
        f2(C2).T
    )  # does f2(C2) here need transpose?

    constC = A + B
    #C = -np.dot(C1, T).dot(C2.T)
    C = -np.dot(h1(C1), T).dot(h2(C2).T)
    tens = constC + C
    return tens * 2


def fgwgrad_partial(alpha, M, C1, C2, T, loss_fun='square_loss'):
    return (1 - alpha) * M + alpha * gwgrad_partial(C1, C2, T, loss_fun)


def partial_fused_gromov_wasserstein(M, C1, C2, p, q, alpha, m=None, G0=None, loss_fun='square_loss', armijo=False, log=False, verbose=False, numItermax=1000, tol=1e-7, stopThr=1e-9, stopThr2=1e-9):
    if m is None:
        # m = np.min((np.sum(p), np.sum(q)))
        raise ValueError("Parameter m is not provided.")
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal to min(|p|_1, |q|_1).")

    if G0 is None:
        G0 = np.outer(p, q)

    nb_dummies = 1
    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1

    if log:
        log = {'err': [], 'loss': []}
    f_val = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(cpt, f_val, 0, 0))
        #print_fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)

    # while err > tol and cpt < numItermax:
    while cpt < numItermax:
        Gprev = np.copy(G0)
        old_fval = f_val

        gradF = fgwgrad_partial(alpha, M, C1, C2, G0, loss_fun)
        gradF_emd = np.zeros(dim_G_extended)
        gradF_emd[:len(p), :len(q)] = gradF
        gradF_emd[-nb_dummies:, -nb_dummies:] = np.max(gradF) * 1e2
        gradF_emd = np.asarray(gradF_emd, dtype=np.float64)

        Gc, logemd = ot.lp.emd(p_extended, q_extended, gradF_emd, numItermax=1000000, log=True)
        if logemd['warning'] is not None:
            raise ValueError("Error in the EMD resolution: try to increase the"
                             " number of dummy points")

        G0 = Gc[:len(p), :len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log['err'].append(err)
            # if verbose:
            #     if cpt % 200 == 0:
            #         print('{:5s}|{:12s}|{:12s}'.format(
            #             'It.', 'Err', 'Loss') + '\n' + '-' * 31)
            #         print('{:5d}|{:8e}|{:8e}'.format(cpt, err,
            #                                          fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)))

        deltaG = G0 - Gprev

        if not armijo:
            a = alpha * gwloss_partial(C1, C2, deltaG, loss_fun)
            b = (1 - alpha) * wloss(M, deltaG) + 2 * alpha * np.sum(gwgrad_partial(C1, C2, deltaG, loss_fun) * 0.5 * Gprev)
            # c = (1 - alpha) * wloss(M, Gprev) + alpha * gwloss_partial(C1, C2, Gprev, loss_fun)
            c = fgwloss_partial(alpha, M, C1, C2, Gprev, loss_fun)

            gamma = ot.optim.solve_1d_linesearch_quad(a, b, c)
            # f_val = a * gamma ** 2 + b * gamma + c
        else:
            def f(x, alpha, M, C1, C2, lossfunc):
                return fgwloss_partial(alpha, M, C1, C2, x, lossfunc)
            xk = Gprev
            pk = deltaG
            gfk = fgwgrad_partial(alpha, M, C1, C2, xk, loss_fun)
            old_val = fgwloss_partial(alpha, M, C1, C2, xk, loss_fun)
            args = (alpha, M, C1, C2, loss_fun)
            gamma, fc, fa = ot.optim.line_search_armijo(f, xk, pk, gfk, old_val, args)
            # f_val = f(xk + gamma * pk, alpha, M, C1, C2, loss_fun)

        if gamma == 0:
            cpt = numItermax
        G0 = Gprev + gamma * deltaG
        f_val = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
        cpt += 1

        # TODO: better stopping criteria?
        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            cpt = numItermax
        if log:
            log['loss'].append(f_val)
        if verbose:
            # if cpt % 20 == 0:
            #     print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            #         'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(cpt, f_val, relative_delta_fval, abs_delta_fval))
            #print_fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)

    if log:
        log['partial_fgw_cost'] = fgwloss_partial(alpha, M, C1, C2, G0, loss_fun)
        return G0[:len(p), :len(q)], log
    else:
        return G0[:len(p), :len(q)]


def partial_pairwise_align(sliceA, sliceB, s, alpha=0.1, armijo=False, dissimilarity='glmpca', use_rep=None, G_init=None, a_distribution=None,
                   b_distribution=None, norm=True, return_obj=False, verbose=True):
    """
    Calculates and returns optimal *partial* alignment of two slices.

    param: sliceA - AnnData object
    param: sliceB - AnnData object
    param: s - Amount of mass to transport; Overlap percentage between the two slices. Note: 0 ≤ s ≤ 1
    param: alpha - Alignment tuning parameter. Note: 0 ≤ alpha ≤ 1
    param: armijo - Whether or not to use armijo (approximate) line search during conditional gradient optimization of Partial-FGW. Default is to use exact line search.
    param: dissimilarity - Expression dissimilarity measure: 'kl' or 'euclidean' or 'glmpca'. Default is glmpca.
    param: use_rep - If none, uses slice.X to calculate dissimilarity between spots, otherwise uses the representation given by slice.obsm[use_rep]
    param: G_init - initial mapping to be used in Partial-FGW OT, otherwise default is uniform mapping
    param: a_distribution - distribution of sliceA spots (1-d numpy array), otherwise default is uniform
    param: b_distribution - distribution of sliceB spots (1-d numpy array), otherwise default is uniform
    param: norm - scales spatial distances such that maximum spatial distance is equal to maximum gene expression dissimilarity
    param: return_obj - returns objective function value if True, nothing if False

    return: pi - partial alignment of spots
    return: log['fgw_dist'] - objective function output of FGW-OT
    """
    m = s
    print("PASTE2 starts...")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance.cdist(sliceB.obsm['spatial'], sliceB.obsm['spatial'])

    # Calculate expression dissimilarity
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, use_rep)), to_dense_array(extract_data_matrix(sliceB, use_rep))
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M = distance.cdist(A_X, B_X)
    elif dissimilarity.lower() == 'gkl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = generalized_kl_divergence(s_A, s_B)
        M /= M[M > 0].max()
        M *= 10
    elif dissimilarity.lower() == 'kl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence(s_A, s_B)
    elif dissimilarity.lower() == 'selection_kl':
        M = high_umi_gene_distance(A_X, B_X, 2000)
    elif dissimilarity.lower() == "pca":
        M = pca_distance(sliceA, sliceB, 2000, 20)
    elif dissimilarity.lower() == 'glmpca':
        M = glmpca_distance(A_X, B_X, latent_dim=50, filter=True, verbose=verbose)
    else:
        print("ERROR")
        exit(1)

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A>0].max()
        #D_A *= 10
        D_A *= M.max()
        D_B /= D_B[D_B>0].max()
        #D_B *= 10
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """
    pi, log = partial_fused_gromov_wasserstein(M, D_A, D_B, a, b, alpha=alpha, m=m, G0=G_init, loss_fun='square_loss', armijo=armijo, log=True, verbose=verbose)

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi



def partial_pairwise_align_histology(sliceA, sliceB, alpha=0.1, s=None, armijo=False, dissimilarity='glmpca', use_rep=None, G_init=None, a_distribution=None,
                   b_distribution=None, norm=True, return_obj=False, verbose=False, **kwargs):
    """
    Optimal partial alignment of two slices using both gene expression and histological image information.

    sliceA, sliceB must be AnnData objects that contain .obsm['rgb'], which stores the RGB value of each spot in the histology image.
    """
    m = s
    print("PASTE2 starts...")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance.cdist(sliceB.obsm['spatial'], sliceB.obsm['spatial'])

    # Calculate expression dissimilarity
    A_X, B_X = to_dense_array(extract_data_matrix(sliceA, use_rep)), to_dense_array(extract_data_matrix(sliceB, use_rep))
    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M_exp = distance.cdist(A_X, B_X)
    elif dissimilarity.lower() == 'kl':
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M_exp = kl_divergence(s_A, s_B)
    elif dissimilarity.lower() == 'glmpca':
        M_exp = glmpca_distance(A_X, B_X, latent_dim=50, filter=True, verbose=verbose)
    else:
        print("ERROR")
        exit(1)

    # Calculate RGB dissimilarity
    # sliceA_rgb = (sliceA.obsm['rgb'] - np.mean(sliceA.obsm['rgb'], axis=0)) / np.std(sliceA.obsm['rgb'], axis=0)
    # sliceB_rgb = (sliceB.obsm['rgb'] - np.mean(sliceB.obsm['rgb'], axis=0)) / np.std(sliceB.obsm['rgb'], axis=0)
    M_rgb = distance.cdist(sliceA.obsm['rgb'], sliceB.obsm['rgb'])
    # M_rgb = distance.cdist(sliceA_rgb, sliceB_rgb)

    # Scale M_exp and M_rgb, obtain M by taking half from each
    M_rgb /= M_rgb[M_rgb > 0].max()
    M_rgb *= M_exp.max()
    # M_exp /= M_exp[M_exp > 0].max()
    # M_rgb /= M_rgb[M_rgb > 0].max()
    M = 0.5 * M_exp + 0.5 * M_rgb

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A>0].max()
        D_A *= M.max()
        D_B /= D_B[D_B>0].max()
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """

    # Run OT
    pi, log = partial_fused_gromov_wasserstein(M, D_A, D_B, a, b, alpha=alpha, m=m, G0=G_init, loss_fun='square_loss', armijo=armijo, log=True, verbose=verbose)

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi


def partial_pairwise_align_given_cost_matrix(sliceA, sliceB, M, s, alpha=0.1, armijo=False, G_init=None, a_distribution=None,
                   b_distribution=None, norm=True, return_obj=False, verbose=False, **kwargs):
    m = s

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]
    # print('Filtered all slices for common genes. There are ' + str(len(common_genes)) + ' common genes.')

    # Calculate spatial distances
    D_A = distance.cdist(sliceA.obsm['spatial'], sliceA.obsm['spatial'])
    D_B = distance.cdist(sliceB.obsm['spatial'], sliceB.obsm['spatial'])

    # init distributions
    if a_distribution is None:
        a = np.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = a_distribution

    if b_distribution is None:
        b = np.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = b_distribution

    if norm:
        D_A /= D_A[D_A > 0].min().min()
        D_B /= D_B[D_B > 0].min().min()

        """
        Code for normalizing distance matrix
        """
        D_A /= D_A[D_A>0].max()
        D_A *= M.max()
        D_B /= D_B[D_B>0].max()
        D_B *= M.max()
        """
        Code for normalizing distance matrix ends
        """

    # Run Partial OT
    pi, log = partial_fused_gromov_wasserstein(M, D_A, D_B, a, b, alpha=alpha, m=m, G0=G_init, loss_fun='square_loss', armijo=armijo, log=True, verbose=verbose)

    if return_obj:
        return pi, log['partial_fgw_cost']
    return pi




