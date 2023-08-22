"""
Python implementation of the generalized PCA for dimension reduction of non-normally distributed data. The original R implementation is at https://github.com/willtownes/glmpca
"""
import numpy as np
from numpy import log
from scipy.special import digamma, polygamma
import statsmodels.genmod.families as smf
from decimal import Decimal


def trigamma(x):
    return polygamma(1, x)


def rowSums(x):
    return x.sum(1)


def rowMeans(x):
    return x.mean(1)


def colSums(x):
    return x.sum(0)


def colMeans(x):
    return x.mean(0)


def colNorms(x):
    """
  compute the L2 norms of columns of an array
  """
    return np.sqrt(colSums(x ** 2))


def ncol(x):
    return x.shape[1]


def nrow(x):
    return x.shape[0]


def crossprod(A, B):
    return (A.T) @ B


def tcrossprod(A, B):
    return A @ (B.T)


def cvec1(n):
    """returns a column vector of ones with length N"""
    return np.ones((n, 1))


def ortho(U, V, A, X=1, G=None, Z=0):
    """
  U is NxL array of cell factors
  V is JxL array of loadings onto genes
  X is NxKo array of cell specific covariates
  A is JxKo array of coefficients of X
  Z is JxKf array of gene specific covariates
  G is NxKf array of coefficients of Z
  assume the data Y is of dimension JxN
  imputed expression: E[Y] = g^{-1}(R) where R = VU'+AX'+ZG'
  """
    if np.all(X == 1): X = cvec1(nrow(U))
    if np.all(Z == 0): Z = np.zeros((nrow(V), 1))
    if np.all(G == 0): G = None
    # we assume A is not null or zero
    # remove correlation between U and A
    # at minimum, this will cause factors to have mean zero
    betax = np.linalg.lstsq(X, U, rcond=None)[0]  # extract coef from linreg
    factors = U - X @ betax  # residuals from linear regression
    A += tcrossprod(V, betax)
    # remove correlation between V and G
    if G is None:
        loadings = V
    else:  # G is not empty
        betaz = np.linalg.lstsq(Z, V, rcond=None)[0]  # extract coef from linreg
        loadings = V - Z @ betaz  # residuals from regression
        G += tcrossprod(factors, betaz)
    # rotate factors to make loadings orthornormal
    loadings, d, Qt = np.linalg.svd(loadings, full_matrices=False)
    factors = tcrossprod(factors, Qt) * d  # d vector broadcasts across cols
    # arrange latent dimensions in decreasing L2 norm
    o = (-colNorms(factors)).argsort()
    factors = factors[:, o]
    loadings = loadings[:, o]
    return {"factors": factors, "loadings": loadings, "coefX": A, "coefZ": G}


def mat_binom_dev(X, P, n):
    """
  binomial deviance for two arrays
  X,P are JxN arrays
  n is vector of length N (same as cols of X,P)
  """
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = X * log(X / (n * P))
    term1 = term1[np.isfinite(term1)].sum()
    # nn= x<n
    nx = n - X
    with np.errstate(divide='ignore', invalid='ignore'):
        term2 = nx * log(nx / (n * (1 - P)))
    term2 = term2[np.isfinite(term2)].sum()
    return 2 * (term1 + term2)


class GlmpcaError(ValueError):
    pass


class GlmpcaFamily(object):
    """thin wrapper around the statsmodels.genmod.families.Family class"""

    # TO DO: would it be better to use inheritance?
    def __init__(self, fam, nb_theta=None, mult_n=None):
        if fam == "poi":
            self.family = smf.Poisson()
        elif fam == "nb":
            if nb_theta is None:
                raise GlmpcaError("Negative binomial dispersion parameter 'nb_theta' must be specified")
            self.family = smf.NegativeBinomial(alpha=1 / nb_theta)
        elif fam in ("mult", "bern"):
            self.family = smf.Binomial()
            if fam == "mult" and mult_n is None:
                raise GlmpcaError("Multinomial sample size parameter vector 'mult_n' must be specified")
        else:
            raise GlmpcaError("unrecognized family type")
        # variance function, determined by GLM family
        vfunc = self.family.variance
        # inverse link func, mu as a function of linear predictor R
        ilfunc = self.family.link.inverse
        # derivative of inverse link function, dmu/dR
        hfunc = self.family.link.inverse_deriv
        self.glmpca_fam = fam
        if fam == "poi":
            def infograd(Y, R):
                M = ilfunc(R)  # ilfunc=exp
                return {"grad": (Y - M), "info": M}
        elif fam == "nb":
            def infograd(Y, R):
                M = ilfunc(R)  # ilfunc=exp
                W = 1 / vfunc(M)
                return {"grad": (Y - M) * W * M, "info": W * (M ** 2)}

            self.nb_theta = nb_theta
        elif fam == "mult":
            def infograd(Y, R):
                P = ilfunc(R)  # ilfunc=expit, P very small probabilities
                return {"grad": Y - (mult_n * P), "info": mult_n * vfunc(P)}

            self.mult_n = mult_n
        elif fam == "bern":
            def infograd(Y, R):
                P = ilfunc(R)
                return {"grad": Y - P, "info": vfunc(P)}
        else:  # this is not actually used but keeping for future reference
            # this is most generic formula for GLM but computationally slow
            raise GlmpcaError("invalid fam")

            def infograd(Y, R):
                M = ilfunc(R)
                W = 1 / vfunc(M)
                H = hfunc(R)
                return {"grad": (Y - M) * W * H, "info": W * (H ** 2)}
        self.infograd = infograd
        # create deviance function
        if fam == "mult":
            def dev_func(Y, R):
                return mat_binom_dev(Y, ilfunc(R), mult_n)
        else:
            def dev_func(Y, R):
                return self.family.deviance(Y, ilfunc(R))
        self.dev_func = dev_func

    def __str__(self):
        return "GlmpcaFamily object of type {}".format(self.glmpca_fam)


def remove_intercept(X):
    cm = colMeans(X)
    try:
        X -= cm
    except TypeError as err:
        if X.dtype != cm.dtype:
            X = X.astype(cm.dtype) - cm
        else:
            raise err
    return X[:, colNorms(X) > 1e-12]


def glmpca_init(Y, fam, sz=None, nb_theta=None):
    """
  create the glmpca_family object and
  initialize the A array (regression coefficients of X)
  Y is the data (JxN array)
  fam is the likelihood
  sz optional vector of size factors, default: sz=colMeans(Y) or colSums(Y)
  sz is ignored unless fam is 'poi' or 'nb'
  """
    if sz is not None and len(sz) != ncol(Y):
        raise GlmpcaError("size factor must have length equal to columns of Y")
    if fam == "mult":
        mult_n = colSums(Y)
    else:
        mult_n = None
    gf = GlmpcaFamily(fam, nb_theta, mult_n)
    if fam in ("poi", "nb"):
        if sz is None: sz = colMeans(Y)  # size factors
        offsets = gf.family.link(sz)
        rfunc = lambda U, V: offsets + tcrossprod(V, U)  # linear predictor
        a1 = gf.family.link(rowSums(Y) / np.sum(sz))
    else:
        rfunc = lambda U, V: tcrossprod(V, U)
        if fam == "mult":  # offsets incorporated via family object
            a1 = gf.family.link(rowSums(Y) / np.sum(mult_n))
        else:  # no offsets (eg, bernoulli)
            a1 = gf.family.link(rowMeans(Y))
    if np.any(np.isinf(a1)):
        raise GlmpcaError("Some rows were all zero, please remove them.")
    return {"gf": gf, "rfunc": rfunc, "intercepts": a1}


def est_nb_theta(y, mu, th):
    """
  given count data y and predicted means mu>0, and a neg binom theta "th"
  use Newton's Method to update theta based on the negative binomial likelihood
  note this uses observed rather than expected information
  regularization:
  let u=log(theta). We use the prior u~N(0,1) as penalty
  equivalently we assume theta~lognormal(0,1) so the mode is at 1 (geometric distr)
  dtheta/du=e^u=theta
  d2theta/du2=theta
  dL/dtheta * dtheta/du
  """
    # n= length(y)
    u = log(th)
    # dL/dtheta*dtheta/du
    score = th * np.sum(digamma(th + y) - digamma(th) + log(th) + 1 - log(th + mu) - (y + th) / (mu + th))
    # d^2L/dtheta^2 * (dtheta/du)^2
    info1 = -(th ** 2) * np.sum(trigamma(th + mu) - trigamma(th) + 1 / th - 2 / (mu + th) + (y + th) / (mu + th) ** 2)
    # dL/dtheta*d^2theta/du^2 = score
    info = info1 - score
    # L2 penalty on u=log(th)
    return np.exp(u + (score - u) / (info + 1))
    # grad= score-u
    # exp(u+sign(grad)*min(maxstep,abs(grad)))


def glmpca(Y, L, fam="poi", ctl={"maxIter": 1000, "eps": 1e-4, "optimizeTheta": True}, penalty=1,
           verbose=False, init={"factors": None, "loadings": None},
           nb_theta=100, X=None, Z=None, sz=None):
    """
    GLM-PCA
    This function implements the GLM-PCA dimensionality reduction method for high-dimensional count data.
    The basic model is R = AX'+ZG'+VU', where E[Y]=M=linkinv(R). Regression coefficients are A and G, latent factors are U, and loadings are V. The objective function being optimized is the deviance between Y and M, plus an L2 (ridge) penalty on U and V. Note that glmpca uses a random initialization, so for fully reproducible results one should set the random seed.
    Parameters
    ----------
    Y: array_like of count data with features as rows and observations as
      columns.
    L: the desired number of latent dimensions (integer).
    fam: string describing the likelihood to use for the data. Possible values include:
    - poi: Poisson
    - nb: negative binomial
    - mult: binomial approximation to multinomial
    - bern: Bernoulli
    ctl: a dictionary of control parameters for optimization. Valid keys:
    - maxIter: an integer, maximum number of iterations
    - eps: a float, maximum relative change in deviance tolerated for convergence
    - optimizeTheta: a bool, indicating if the overdispersion parameter of the NB
      distribution is optimized (default), or fixed to the value provided in nb_theta.
    penalty: the L2 penalty for the latent factors (default = 1).
      Regression coefficients are not penalized.
    verbose: logical value indicating whether the current deviance should
      be printed after each iteration (default = False).
    init: a dictionary containing initial estimates for the factors (U) and
      loadings (V) matrices.
    nb_theta: negative binomial dispersion parameter. Smaller values mean more dispersion
      if nb_theta goes to infinity, this is equivalent to Poisson
      Note that the alpha in the statsmodels package is 1/nb_theta.
      If ctl["optimizeTheta"] is True, this is used as initial value for optimization
    X: array_like of column (observations) covariates. Any column with all
      same values (eg. 1 for intercept) will be removed. This is because we force
      the intercept and want to avoid collinearity.
    Z: array_like of row (feature) covariates, usually not needed.
    sz: numeric vector of size factors to use in place of total counts.
    Returns
    -------
    A dictionary with the following elements
    - factors: an array U whose rows match the columns (observations) of Y. It is analogous to the principal components in PCA. Each column of the factors array is a different latent dimension.
    - loadings: an array V whose rows match the rows (features/dimensions) of Y. It is analogous to loadings in PCA. Each column of the loadings array is a different latent dimension.
    - coefX: an array A of coefficients for the observation-specific covariates array X. Each row of coefX corresponds to a row of Y and each column corresponds to a column of X. The first column of coefX contains feature-specific intercepts which are included by default.
    - coefZ: a array G of coefficients for the feature-specific covariates array Z. Each row of coefZ corresponds to a column of Y and each column corresponds to a column of Z. By default no such covariates are included and this is returned as None.
    - dev: a vector of deviance values. The length of the vector is the number of iterations it took for GLM-PCA's optimizer to converge. The deviance should generally decrease over time. If it fluctuates wildly, this often indicates numerical instability, which can be improved by increasing the penalty parameter.
    - glmpca_family: an object of class GlmpcaFamily. This is a minor wrapper to the family object used by the statsmodels package for fitting standard GLMs. It contains various internal functions and parameters needed to optimize the GLM-PCA objective function. For the negative binomial case, it also contains the final estimated value of the dispersion parameter nb_theta.
    Examples
    -------
    1) create a simple dataset with two clusters and visualize the latent structure
    # >>> from numpy import array,exp,random,repeat
    # >>> from matplotlib.pyplot import scatter
    # >>> from glmpca import glmpca
    # >>> mu= exp(random.randn(20,100))
    # >>> mu[range(10),:] *= exp(random.randn(100))
    # >>> clust= repeat(["red","black"],10)
    # >>> Y= random.poisson(mu)
    # >>> res= glmpca(Y.T, 2)
    # >>> factors= res["factors"]
    # >>> scatter(factors[:,0],factors[:,1],c=clust)
    References
    ----------
    .. [1] Townes FW, Hicks SC, Aryee MJ, and Irizarry RA. "Feature selection and dimension reduction for single-cell RNA-seq based on a multinomial model", biorXiv, 2019. https://www.biorxiv.org/content/10.1101/574574v1
    .. [2] Townes FW. "Generalized principal component analysis", arXiv, 2019. https://arxiv.org/abs/1907.02647
  """
    # For negative binomial, convergence only works if starting with nb_theta large
    Y = np.array(Y)
    if fam not in ("poi", "nb", "mult", "bern"): raise GlmpcaError("invalid fam")
    J, N = Y.shape
    # sanity check inputs
    if fam in ("poi", "nb", "mult", "bern") and np.min(Y) < 0:
        raise GlmpcaError("for count data, the minimum value must be >=0")
    if fam == "bern" and np.max(Y) > 1:
        raise GlmpcaError("for Bernoulli model, the maximum value must be <=1")

    # preprocess covariates and set updateable indices
    if X is not None:
        if nrow(X) != ncol(Y):
            raise GlmpcaError("X rows must match columns of Y")
        # we force an intercept, so remove it from X to prevent collinearity
        X = remove_intercept(X)
    else:
        X = np.zeros((N, 0))  # empty array to prevent dim mismatch errors with hstack later
    Ko = ncol(X) + 1
    if Z is not None:
        if nrow(Z) != nrow(Y):
            raise GlmpcaError("Z rows must match rows of Y")
    else:
        Z = np.zeros((J, 0))  # empty array to prevent dim mismatch errors with hstack later
    Kf = ncol(Z)
    lid = (Ko + Kf) + np.array(range(L))
    uid = Ko + np.array(range(Kf + L))
    vid = np.concatenate((np.array(range(Ko)), lid))
    Ku = len(uid)
    Kv = len(vid)

    # create GlmpcaFamily object
    gnt = glmpca_init(Y, fam, sz, nb_theta)
    gf = gnt["gf"]
    rfunc = gnt["rfunc"]
    a1 = gnt["intercepts"]

    # initialize U,V, with row-specific intercept terms
    U = np.hstack((cvec1(N), X, np.random.randn(N, Ku) * 1e-5 / Ku))
    if init["factors"] is not None:
        L0 = np.min([L, ncol(init["factors"])])
        U[:, (Ko + Kf) + np.array(range(L0))] = init["factors"][:, range(L0)]
    # a1 = naive MLE for gene intercept only, must convert to column vector first with [:,None]
    V = np.hstack((a1[:, None], np.random.randn(J, (Ko - 1)) * 1e-5 / Kv))
    # note in the above line the randn can be an empty array if Ko=1, which is OK!
    V = np.hstack((V, Z, np.random.randn(J, L) * 1e-5 / Kv))
    if init["loadings"] is not None:
        L0 = np.min([L, ncol(init["loadings"])])
        V[:, (Ko + Kf) + np.array(range(L0))] = init["loadings"][:, range(L0)]

    # run optimization
    dev = np.repeat(np.nan, ctl["maxIter"])
    for t in range(ctl["maxIter"]):
        dev[t] = gf.dev_func(Y, rfunc(U, V))
        if not np.isfinite(dev[t]):
            raise GlmpcaError(
                "Numerical divergence (deviance no longer finite), try increasing the penalty to improve stability of optimization.")
        if t > 4 and np.abs(dev[t] - dev[t - 1]) / (0.1 + np.abs(dev[t - 1])) < ctl["eps"]:
            break
        if verbose:
            msg = "Iteration: {:d} | deviance={:.4E}".format(t, Decimal(dev[t]))
            if fam == "nb": msg += " | nb_theta: {:.3E}".format(nb_theta)
            print(msg)

        # (k in lid) ensures no penalty on regression coefficients:
        for k in vid:
            ig = gf.infograd(Y, rfunc(U, V))
            grads = ig["grad"] @ U[:, k] - penalty * V[:, k] * (k in lid)
            infos = ig["info"] @ (U[:, k] ** 2) + penalty * (k in lid)
            V[:, k] += grads / infos
        for k in uid:
            ig = gf.infograd(Y, rfunc(U, V))
            grads = crossprod(ig["grad"], V[:, k]) - penalty * U[:, k] * (k in lid)
            infos = crossprod(ig["info"], V[:, k] ** 2) + penalty * (k in lid)
            U[:, k] += grads / infos
        if fam == "nb":
            if ctl["optimizeTheta"]:
                nb_theta = est_nb_theta(Y, gf.family.link.inverse(rfunc(U, V)), nb_theta)
            gf = GlmpcaFamily(fam, nb_theta)
    # postprocessing: include row and column labels for regression coefficients
    if ncol(Z) == 0:
        G = None
    else:
        G = U[:, Ko + np.array(range(Kf))]
    X = np.hstack((cvec1(N), X))
    A = V[:, range(Ko)]
    res = ortho(U[:, lid], V[:, lid], A, X=X, G=G, Z=Z)
    res["dev"] = dev[range(t + 1)]
    res["glmpca_family"] = gf
    return res


if __name__ == "__main__":
    from numpy import exp, random, repeat

    mu = exp(random.randn(20, 100))
    mu[range(10), :] *= exp(random.randn(100))
    clust = repeat(["red", "black"], 10)
    Y = random.poisson(mu)
    res = glmpca(Y.T, 2, fam="nb", verbose=True)
    factors = res["factors"]
    # from matplotlib.pyplot import scatter
    # %pylab
    # scatter(factors[:,0],factors[:,1],c=clust)
