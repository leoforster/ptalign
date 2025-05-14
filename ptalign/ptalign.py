import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from scipy.spatial.distance import cdist, euclidean
from scipy.stats import pearsonr, gaussian_kde

#############################################################################
### UTILITY FUNCTIONS
#############################################################################

def to_lognorm_counts(counts, log=True):
    """
    Return a sizefactor-scaled and optionally log-normalized counts matrix
    """
    size_factors = counts.sum() / counts.sum().mean()
    scaled = counts / size_factors
    return np.log1p(scaled) if log else scaled

def plot_correlation_matrices(cellcorr, cellcorr_s, cellpt):
    """
    Plots raw and scaled correlation matrices
    """
    f = plt.figure(figsize=(10, 4))

    grid = f.add_gridspec(ncols=5, nrows=5, 
                          width_ratios=[1, 0.05, 0.1, 1, 0.05], wspace=0.1,
                          height_ratios=[1, 0.3, 1, 0.1, 0.1], hspace=0)

    ax1 = f.add_subplot(grid[0:3, 0]) # raw correlation profiles
    raw = ax1.imshow(cellcorr.T, cmap='RdBu_r', aspect='auto')
    rawcb = f.colorbar(raw, cax=f.add_subplot(grid[0, 1]))
    rawcb.set_label('Correlation', fontsize=10, rotation=90, labelpad=-20)
    rawcb.set_ticks([cellcorr.max().max(), cellcorr.min().min()])
    rawcb.set_ticklabels([round(cellcorr.max().max(), 2), round(cellcorr.min().min(), 2)])

    ax2 = f.add_subplot(grid[0:3, 3]) # scaled correlation profiles
    sca = ax2.imshow(cellcorr_s.T, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    scacb = f.colorbar(sca, cax=f.add_subplot(grid[0, 4]))
    scacb.set_label('Correlation', fontsize=10, rotation=90, labelpad=-10)
    scacb.set_ticks([cellcorr_s.max().max(), cellcorr_s.min().min()])
    scacb.set_ticklabels([round(cellcorr_s.max().max(), 2), round(cellcorr_s.min().min(), 2)])

    ax3 = f.add_subplot(grid[4, 3], sharex=ax2) # cell ptalign bar
    pta = ax3.imshow(cellpt.values.reshape(-1, 1).T, aspect='auto')
    ptacb = f.colorbar(pta, cax=f.add_subplot(grid[2, 4]))
    ptacb.set_label('Pseudotime', fontsize=10, rotation=90, labelpad=-10)
    ptacb.set_ticks([cellpt.max(), cellpt.min()])
    ptacb.set_ticklabels([round(cellpt.max(), 2), round(cellpt.min(), 2)])

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        
    ax1.set_xlabel('Query cells')
    ax3.set_xlabel('Query cells (aligned pseudotime)')

    ax1.set_ylabel('Reference pseudotime')
    ax1.set_title('Raw cell similarity correlation profiles', fontsize=10)
    ax2.set_title('Scaled cell similarity correlation profiles', fontsize=10)

    f.align_labels()

    plt.show()

def plot_ptalign_metrics(cellpt, permscore, ptalignscore, permpval, dtcorr, route):
    """
    Plots ptalign pseudotime cell density, permutation pvalue, and DTW with traceback
    """
    f = plt.figure(figsize=(10, 4))

    grid = f.add_gridspec(ncols=10, nrows=8, 
                          width_ratios=[1]*10, wspace=0,
                          height_ratios=[1]*8, hspace=0)

    ax1 = f.add_subplot(grid[0:3, 0:4]) # cell pseudotime density
    kde = gaussian_kde(cellpt)
    ax1.plot(cellpt, kde(cellpt), lw=3, c='cornflowerblue')
    ax1.set_xlabel('Query ptalign pseudotime')
    ax1.set_ylabel('Cell density')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = f.add_subplot(grid[5:, 0:4]) # permutation pvalue
    if len(permscore) >= 3:
        kde = gaussian_kde(permscore)
        xs = np.linspace(min(permscore), max(permscore), 100)
        ax2.plot(xs, kde(xs), lw=3, c='cornflowerblue', label='Permutations (n=%s)' %len(permscore))
        ax2.plot((ptalignscore, ptalignscore), (min(kde(xs)), max(kde(xs))), 
                 ls=':', lw=3, c='lightcoral', label='ptalign')
        ax2.text(ptalignscore, max(kde(xs)), 'p=%s' %(permpval), ha='center', va='bottom')
        ax2.set_xlabel('DTW traceback score')
        ax2.set_ylabel('Permutation\nscore density')
        ax2.legend(frameon=False, loc='lower left')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    else:
        ax2.set_xticks([])
        ax2.text(0.5, 0.5, 'Insufficient permutations\nspecified', transform=ax2.transAxes, ha='center', va='center')

    ax3 = f.add_subplot(grid[1:7, 6:9]) # DTW traceback
    ax3.imshow(dtcorr, cmap='RdBu_r', aspect='auto')
    x, y = zip(*route)
    ax3.plot((0, route[0][1]), (0, route[0][0]), ls='--', lw=2, c='gray')
    ax3.plot([i for i in y], [i for i in x], c='k', lw=6)
    ax3.plot([i for i in y], [i for i in x], c='gray', lw=3)
    ax3.set_xticks([i for i in range(dtcorr.shape[1])])
    ax3.set_yticks([i for i in range(dtcorr.shape[0])])
    ax3.set_xticklabels(dtcorr.columns)
    ax3.set_yticklabels(dtcorr.index)
    ax3.set_xlabel('Query ptalign pseudotime bin')
    ax3.set_ylabel('Reference pseudotime bin')
    ax3.set_title('Query-Reference DTW with traceback', fontsize=10)

    for ax in [ax1, ax2]:
        ax.set_yticks([])

    plt.show()

#############################################################################
### PTALIGN HELPER FUNCTIONS
#############################################################################

def expr_matched_permuted_genesets(raw_refcounts, genepool, geneset,
                                   minmean=0.05, bins=10, scale_counts=True,
                                   min_genes_per_bin=3, permutations=100):
    """
    Select permuted gene sets matched by expression distribution of a target set.
    """
    genemean = pd.DataFrame(index=raw_refcounts.index)

    if scale_counts:
        raw_refcounts = to_lognorm_counts(raw_refcounts, log=True)

    genemean['refmean'] = raw_refcounts.mean(1) # mean expression per gene in reference
    genemean = genemean.reindex(genepool).dropna()
    genemean = genemean[genemean['refmean'] > minmean]

    # bin genes by mean expression in reference
    genemean['refbins'] = pd.cut(genemean['refmean'], bins=bins, labels=range(bins))

    # frequency per bin from ptalign gene set
    samplesize = genemean.reindex(geneset)['refbins'].value_counts().to_dict()

    # require minimum number of genes per bin
    genemean = genemean.drop(index=[i for i in geneset if i in genemean.index])
    for idx, i in (genemean['refbins'].value_counts() < min_genes_per_bin).items():
        if i and samplesize[idx] > 0:
            print('permutation geneset expression bin %s has < %s cells, ignoring %s genes from there' 
                  %(idx, min_genes_per_bin, samplesize[idx]))
            del samplesize[idx]

    # populate permuted gene sets
    permgenes = {i:[] for i in range(permutations)}
    for i in sorted(samplesize.keys()):
        p = genemean[genemean['refbins'] == i]
        for n in range(permutations):
            permgenes[n] += p.sample(samplesize[i]).index.tolist()

    return permgenes

def binned_ptime_dtw(refdtw, refbin,
                     tumdtw, tumbin,
                     scale_counts=True):
    """
    Compute pairwise Pearson correlation between cells in binned reference and query pseudotimes.
    """
    refdtw['ptbin'] = refbin.dropna()
    tumdtw['ptbin'] = tumbin.dropna()
    assert (refdtw.columns == tumdtw.columns).all()

    if scale_counts:
        refdtw = to_lognorm_counts(refdtw.groupby('ptbin').sum(), log=True)
        tumdtw = to_lognorm_counts(tumdtw.groupby('ptbin').sum(), log=True)
    else:
        # mean over sum since lognorm above accounts for cell number (sum doesnt)
        refdtw = refdtw.groupby('ptbin').mean()
        tumdtw = tumdtw.groupby('ptbin').mean()

    tumdtw = tumdtw[tumdtw.sum(1) > 0] # bins where no cells reside
    # fill NA for genes with no expression in a bin, but expr elsewhere so dont ignore
    tumdtw = tumdtw.fillna(0)
    refdtw = refdtw.fillna(0)

    # DTW
    dtcorr = pd.DataFrame(index=refdtw.index, columns=tumdtw.index)
    for i in dtcorr.index:
        for j in dtcorr.columns:
            dtcorr.at[i, j] = pearsonr(refdtw.loc[i], tumdtw.loc[j])[0]
    dtcorr = dtcorr.astype('float64')

    return dtcorr

def dtw_path_by_dp(dtwmat):
    """
    Fill DTW traceback matrix using dynamic programming,
    modified from https://karthikkaranth.me/blog/implementing-seam-carving-with-python/

    Returns
    -------
    M : np.ndarray
        DP matrix of accumulated scores.
    b : np.ndarray
        Traceback pointer matrix (0=up,1=left,2=diag).
    """
    M = dtwmat.values.copy()
    b = np.zeros_like(M, dtype=int)
    b[0] = 1 # define traceback on first row

    r, c = M.shape
    for i in range(1, r):
        for j in range(1, c):
            up = M[i - 1, j]
            le = M[i, j - 1]
            dg = M[i - 1, j - 1]
            term = max(up, le, dg)

            M[i, j] += term
            b[i, j] = 0 if term == up else (1 if term == le else 2)

    return M, b

def dtw_path_traceback(tb):
    """
    Traverse traceback matrix to find maximizing path from bottom-right to (0,0)

    Parameters
    ----------
    tb : np.ndarray
        Traceback pointer matrix from dtw_path_by_dp.

    Returns
    -------
    list of tuple
        Coordinates (row, col) along the maximizing DTW path.
    """
    rend = tb.shape[0] - 1
    cend = tb.shape[1] - 1
    trace = []
    while True:
        trace.append((rend, cend)) # start at (rend, cend), ie. bottom-right
        if rend == 0 and cend == 0: # reached (0,0)
            break
        if tb[rend, cend] == 0:
            rend -= 1
        elif tb[rend, cend] == 1:
            cend -= 1
        elif tb[rend, cend] == 2:
            rend -= 1
            cend -= 1

    return trace

#############################################################################
### PTIME ASSIGNMENT FUNCTIONS
#############################################################################

def train_reference_mlp(cellcorr_s, refpt,
                        n_pt_bins, alphas=None,
                        gridcv=5, gridcores=-1):
    """
    Grid search MLP hyperparameters to return a pseudotime-predictive MLP regressor

    Parameters
    ----------
    cellcorr_s : pd.DataFrame
        Scaled correlation profiles (cells x bins).
    refpt : pd.Series
        Pseudotime values for reference cells.
    n_pt_bins : int
        Number of pseudotime bins (used to define network architecture).
    alphas : list of float, optional
        Grid search L2 regularization strengths; defaults to log-uniform grid.
    gridcv : int, default=5
        Number of cross-validation folds for grid search.
    gridcores : int, default=-1
        Number of parallel jobs for grid search.

    Returns
    -------
    MLPRegressor
        Fitted regressor with best-found hyperparameters.
    """
    assert (cellcorr_s.index == refpt.index).all()

    # 2-layer MLP regressor with decaying hidden layer sizes
    reg = MLPRegressor(hidden_layer_sizes=(int(n_pt_bins/2), int(np.sqrt(n_pt_bins))),
                       solver='adam',
                       learning_rate_init=0.001,
                       activation='relu', # best in internal testing
                       batch_size='auto',
                       max_iter=200,
                       shuffle=True,
                       random_state=1)

    # define grid search over MLP alpha parameter, representing L2 regularization term
    if alphas is None: alphas = [np.exp(i) for i in np.linspace(np.log(10), np.log(1e-3), 33)]
    grid = GridSearchCV(reg, {'alpha':alphas}, refit=True,
                        scoring='neg_mean_squared_error',
                        n_jobs=gridcores, cv=gridcv)

    # forego train-test split for grid-search internal CV
    gridres = grid.fit(cellcorr_s, refpt)

    return gridres.best_estimator_

#############################################################################
### PTALIGN CORE FUNCTIONS
#############################################################################

def ptalign_correlations(refpt, refcounts, tumcounts,
                         n_pt_bins=50, 
                         scale_counts=True,
                         squish_correlation_rowmean=True,
                         cell_min_maxcorr=0.2):
    """
    Compute normal and scaled correlation matrix between reference and query (tumor)

    Parameters
    ----------
    refpt : pd.Series
        Pseudotime values for reference cells.
    refcounts : pd.DataFrame
        Reference counts (genes x cells).
    tumcounts : pd.DataFrame
        Query counts (genes x cells).
    n_pt_bins : int, default=50
        Number of bins to discretize reference pseudotime for correlations.
    scale_counts : bool, default=True
        Whether to log-normalize binned counts.
    cell_min_maxcorr : float, default=0.2
        Minimum max correlation to retain a query cell.
    squish_correlation_rowmean : bool, default=True
        Whether to subtract squared mean from each row before scaling.

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple of (raw_correlation, scaled_correlation).
    """
    # bin reference cells in ptime and average expression per bin
    ref_ptbins, binstarts = pd.qcut(refpt.sort_values(), n_pt_bins, labels=range(n_pt_bins), retbins=True)
    ref_bincounts = pd.DataFrame(index=refcounts.index)
    for i in range(n_pt_bins):
        ref_bincounts[i] = refcounts[ref_ptbins[ref_ptbins == i].index.tolist()].mean(1)

    # lognorm on binned counts
    if scale_counts:
        ref_nbincounts = to_lognorm_counts(ref_bincounts, log=True)
        tumcounts = to_lognorm_counts(tumcounts, log=True)
    else:
        ref_nbincounts = ref_bincounts

    # pearson correlation using scipy cdist (its very fast)
    corrdist = 1 - cdist(tumcounts.T, ref_nbincounts.T, metric='correlation') # cells in rows
    cellcorr = pd.DataFrame(corrdist, index=tumcounts.columns)

    # filter cells where maximum correlation value is below cell_min_maxcorr
    cellcorr = cellcorr[cellcorr.max(1) > cell_min_maxcorr]

    # center row by square of mean
    if squish_correlation_rowmean:
        corrmod = cellcorr - np.square(cellcorr.mean())
    else:
        corrmod = cellcorr

    # scale correlations
    corrmod_s = corrmod.subtract(corrmod.min(1), 0)
    corrmod_s = corrmod_s.divide(corrmod_s.max(1), 0)

    return cellcorr, corrmod_s

def ptalign_ordercells(cellcorr, cellcorr_s, refpt, keyargs):
    """
    Predict cell's pseudotime (clipped to [0,1]) from their correlation shape using pretrained MLP regressor
    """
    assert 'mlpreg' in keyargs.keys() # for permutation compatibility
    cellpt = pd.Series(keyargs['mlpreg'].predict(cellcorr_s), index=cellcorr.index).clip(0, 1).sort_values()

    return cellpt

def ptalign_dtwscore(refpt, refcounts, cellpt, tumcounts,
                     cdelta=None, n_corr_bins=20,
                     min_dtw_bin_cells=5,
                     scale_counts=True):
    """
    Compute DTW between reference and query from pseudotimes and counts matrices,
    also processes DTW scoring by correlation-maximizing traceback

    Returns
    -------
    dtcorr : pd.DataFrame
        DTW matrix of bin-wise expression correlation.
    route : list of tuple
        Score-maximising path through DTW matrix.
    dtw_score : float
        Accumulated DTW score at bottom-right of DP matrix.
    """
    # define ptime bin edges by ref
    refbin, binedge = pd.cut(refpt, bins=n_corr_bins,
                             labels=range(n_corr_bins), retbins=True)
    tumbin = pd.cut(cellpt, bins=binedge, labels=range(n_corr_bins))

    # apply binsize thresholds
    binfail = [idx for idx, i in (tumbin.value_counts() < min_dtw_bin_cells).items() if i]
    tumbin = tumbin[~tumbin.isin(binfail)]

    # DTW
    dtcorr = binned_ptime_dtw(refcounts.T, refbin,
                              tumcounts.T, tumbin,
                              scale_counts=scale_counts)

    # weight by delta-corr, improves detection of spurious signals in permuted DTWs
    if not cdelta is None:
        bindelta = cdelta.reindex(tumbin.index).groupby(tumbin).mean().dropna()
        dtcorr = dtcorr.multiply(bindelta)

    # DTW path by DP and traceback score
    dpdist, traceback = dtw_path_by_dp(dtcorr)
    route = dtw_path_traceback(traceback)
    dtw_score = dpdist[-1, -1]

    return dtcorr, route, dtw_score

def ptalign_permutation_step(permcounts_ref, refpt, permcounts_tum,
                             n_pt_bins, n_corr_bins, cell_min_maxcorr,
                             num_cores, scale_counts, scale_dtw, retain, keyargs):
    """
    Assess ptalign pseudotime for permuted gene set, reporting DTW and DTW score.
    Function arguments are inherited from parent permute_ptalign function.
    """
    if keyargs:
        if 'min_dtw_bin_cells' in keyargs: min_dtw_bin_cells = keyargs['min_dtw_bin_cells']
        if 'squish_correlation_rowmean' in keyargs: squish_correlation_rowmean = keyargs['squish_correlation_rowmean']
    else:
        min_dtw_bin_cells = 5
        squish_correlation_rowmean = True

    permres = {}

    # compute base ptalign correlation matrices
    cellcorr, cellcorr_s = ptalign_correlations(refpt, permcounts_ref, permcounts_tum,
                                                n_pt_bins=n_pt_bins, 
                                                scale_counts=scale_counts,
                                                squish_correlation_rowmean=squish_correlation_rowmean,
                                                cell_min_maxcorr=cell_min_maxcorr)

    # derive ptalign pseudotime from pretrained reference MLP
    cellpt = ptalign_ordercells(cellcorr, cellcorr_s, refpt, keyargs)

    # filter tumor table since cells could be removed in above step
    permcounts_tum = permcounts_tum[cellpt.index]
    permcounts_tum = permcounts_tum[permcounts_tum.sum(1) > 0]
    permcounts_ref = permcounts_ref.reindex(permcounts_tum.index)
    permcounts_ref = permcounts_ref[refpt.index]

    # score ptalign
    cdelta = cellcorr.max(1) - cellcorr.min(1) if scale_dtw else None
    dtcorr, route, perm_score = ptalign_dtwscore(refpt, permcounts_ref,
                                                 cellpt, permcounts_tum,
                                                 n_corr_bins=n_corr_bins,
                                                 min_dtw_bin_cells=min_dtw_bin_cells,
                                                 scale_counts=scale_counts,
                                                 cdelta=cdelta)

    # save outputs
    if 'all' in retain:
        retain = ['corrmat', 'scaled_corrmat', 'cellpt', 'dtw', 'route']

    if 'corrmat' in retain: permres['corrmat'] = cellcorr
    if 'scaled_corrmat' in retain: permres['scaled_corrmat'] = cellcorr_s
    if 'cellpt' in retain: permres['cellpt'] = cellpt
    if 'dtw' in retain: permres['dtw'] = dtcorr
    if 'route' in retain: permres['route'] = route

    permres['dpscore'] = perm_score

    return permres

def permute_ptalign(raw_refcounts, refpt, raw_tumcounts,
                    n_pt_bins, n_corr_bins, ptalign_geneset,
                    permutations=100, num_cores=22,
                    scale_counts=True, scale_dtw=True, cell_min_maxcorr=-1,
                    retain=['corrmat', 'cellpt', 'dtw', 'route'],
                    keyargs=None):
    """
    Manage ptalign permutation steps through gene set selection and ptalign pseudotime derivation
    
    Parameters
    ----------
    raw_refcounts : pd.DataFrame
        Raw reference counts (genes x cells).
    refpt : pd.Series
        Pseudotime values for reference cells.
    raw_tumcounts : pd.DataFrame
        Raw query counts (genes x cells).
    n_pt_bins : int
        Number of bins to discretize reference pseudotime for correlations.
    n_corr_bins : int
        Number of pseudotime bins for DTW.
    ptalign_geneset : list of str
        Original geneset to align with.
    permutations : int, default=100
        Number of permuted genesets to test via DTW.
    num_cores : int, default=1
        Number of parallel processes to use.
    scale_counts : bool, default=True
        Whether to normalize counts before correlation.
    scale_dtw : bool, default=True
        Whether to weight DTW by correlation range.
    cell_min_maxcorr : float, default=-1
        Minimum max correlation to retain a query cell.
    retain : list of str
        Result types to keep for each permutation.
    keyargs : dict, optional
        Additional arguments (must include 'mlpreg').

    Returns
    -------
    list of dict
        List with one result dictionary for each permutation.
    """
    # get permutation genesets
    raw_refcounts = raw_refcounts[raw_refcounts.sum(1) > 0]
    raw_tumcounts = raw_tumcounts[raw_tumcounts.sum(1) > 0]
    both_expressed = list(set(raw_refcounts.index).intersection(set(raw_tumcounts.index)))

    permgenes = expr_matched_permuted_genesets(raw_refcounts, both_expressed, ptalign_geneset,
                                               minmean=0.05, bins=10, scale_counts=scale_counts,
                                               min_genes_per_bin=3, permutations=permutations)

    # populate count tables
    # note: if permutations is large, might be better to pass raw_* matrices and filter per thread
    permcounts = {i:(raw_refcounts.reindex(permgenes[i]).reset_index(drop=True),
                     raw_tumcounts.reindex(permgenes[i]).reset_index(drop=True)) for i in range(permutations)}

    # parallel execution via joblib, running ptalign_permutation_step function
    permres = Parallel(n_jobs=num_cores)(
                        delayed(ptalign_permutation_step)(
                                    permcounts[i][0], refpt, permcounts[i][1],
                                    n_pt_bins, n_corr_bins, cell_min_maxcorr,
                                    1, scale_counts, scale_dtw, retain, keyargs
                        ) for i in range(permutations)
                      )

    return permres

#############################################################################
### PSEUDOTIME ALIGNMENT
#############################################################################

def pseudotime_alignment(refpt, raw_refcounts, raw_tumcounts, geneset,
                         n_pt_bins=50, num_cores=22,
                         scale_counts=True,
                         scale_dtw=True,
                         squish_correlation_rowmean=True,
                         alphas=None,
                         gridcv=5,
                         cell_min_geneset_genes=10,
                         cell_min_maxcorr=0.2,
                         min_dtw_bin_cells=5,
                         makeplots=True, n_corr_bins=20,
                         permutations=100, retain=['all']):
    """
    ptalign entry function which manages query-reference correlation matrix, training ptime assignment network,
    DTW computation and scoring, as well as gene set permutation, alignment, and resulting permuted DTW scores.
    
    Parameters
    ----------
    refpt : pd.Series
        Pseudotime values for reference cells.
    raw_refcounts : pd.DataFrame
        Raw reference counts (genes x cells).
    raw_tumcounts : pd.DataFrame
        Raw query counts (genes x cells).
    geneset : list of str
        Genes to use for alignment.
    n_pt_bins : int, default=50
        Number of bins to discretize reference pseudotime for correlations.
    num_cores : int, default=1
        Number of parallel processes to use.
    scale_counts : bool, default=True
        Whether to normalize counts before correlation.
    scale_dtw : bool, default=True
        Whether to weight DTW by correlation range.
    squish_correlation_rowmean : bool, default=True
        Whether to subtract squared mean from each row before scaling correlations.
    alphas : list of float, optional
        Regularization parameters for MLP grid search.
    gridcv : int, default=5
        Cross-validation folds for MLP hyperparameter selection.
    cell_min_geneset_genes : int, default=10
        Minimum number of expressed geneset genes to keep query cell.
    cell_min_maxcorr : float, default=0.2
        Minimum max correlation to retain a query cell.
    min_dtw_bin_cells : int, default=5
        Minimum cells to keep a DTW pseudotime bin.
    makeplots : bool, default=False
        Whether to generate diagnostic plots.
    n_corr_bins : int, default=20
        Number of pseudotime bins for DTW.
    permutations : int, default=0
        Number of permuted genesets to test via DTW.
    retain : list of str, defaul=['all']
        Result types to keep in output dictionary.

    Returns
    -------
    dict or (dict, list of dict)
        If permutations=0, returns a single result dict;
        otherwise returns (result dict, list of permutation result dicts).
    
    This function returns a dictionary or a tuple of dictionaries, depending if permutations=0 or >0. In the former
    case, the dictionary keys comprise all or a subset of ['corrmat', 'scaled_corrmat', 'cellpt', 'dtw', 'route', 'dpscore'],
    depending on the values passed in the retain argument. If permutations is >0, the first dictionary holds the results
    as described above, while the second contains an array of similarly-formatted dictionaries, each holding the 
    results for an individual gene set permutation. Which of these values are populated can be controlled via the
    retain argument, which defaults to 'all' but can specify a list of one or more keys to retain while excluding
    the others.
    Among these output dictionaries, the corrmat key holds the raw correlation matrix between the binned reference
    and individual query cells. The cellpt key contains a pandas Series indexed by query cell names with ptalign
    pseudotime as values. The dtw key references the query-reference DTW, while route and dpscore keys contain
    the coordinates of the maximizing traceback route and its score, respectively.
    """
    print('step0: preprocessing counts tables')
    assert raw_refcounts.index.isin(geneset).sum() == len(geneset)

    tumcounts = raw_tumcounts.reindex(geneset).dropna()
    tumcounts = tumcounts.loc[:, tumcounts.sum() > cell_min_geneset_genes]

    if (tumcounts.sum(1) == 0).sum() > 0: # gene is not expressed in tumors
        tumcounts = tumcounts[tumcounts.sum(1) > 0]

    if tumcounts.shape != (len(geneset), raw_tumcounts.shape[1]):
        print('       %s genes (%.1f%%) and %s cells (%.1f%%) detected in raw_tumcounts' 
              %(tumcounts.shape[0],
                ((tumcounts.shape[0]/len(geneset))*100),
                tumcounts.shape[1],
                ((tumcounts.shape[1]/raw_tumcounts.shape[1])*100)))

    refcounts = raw_refcounts.reindex(tumcounts.index).dropna()
    refcounts = refcounts[refpt.index]

    assert refcounts.shape[0] == tumcounts.shape[0]
    assert len(refpt) == refcounts.shape[1]

    print('step1: computing correlation matrix')
    # process keyword arguments separate as same function is used by pool.starmap
    keyargs = {'squish_correlation_rowmean':squish_correlation_rowmean,
               'min_dtw_bin_cells':min_dtw_bin_cells,
               'makeplots':makeplots}

    ptalign = {}

    # compute base ptalign correlation matrices
    cellcorr, cellcorr_s = ptalign_correlations(refpt, refcounts, tumcounts,
                                                n_pt_bins=n_pt_bins,
                                                scale_counts=scale_counts,
                                                squish_correlation_rowmean=squish_correlation_rowmean,
                                                cell_min_maxcorr=cell_min_maxcorr)

    print('step2: deriving cell ptimes')
    # compute reference-reference correlation matrix for training MLP regressor
    _, refref_s = ptalign_correlations(refpt, refcounts, refcounts,
                                       n_pt_bins=n_pt_bins,
                                       scale_counts=scale_counts,
                                       squish_correlation_rowmean=False,
                                       cell_min_maxcorr=-1)

    # train ptime prediction from correlation dynamics on reference-reference correlations
    mlpreg = train_reference_mlp(refref_s, refpt,
                                 n_pt_bins, alphas=alphas,
                                 gridcv=gridcv, gridcores=num_cores)
    keyargs['mlpreg'] = mlpreg

    # derive pseudotimes for query cells
    cellpt = ptalign_ordercells(cellcorr, cellcorr_s, refpt, keyargs)

    cellcorr = cellcorr.reindex(cellpt.index)
    cellcorr_s = cellcorr_s.reindex(cellpt.index)

    print('step3: run alignment traceback')
    # compute DTW and related metrics
    cdelta = cellcorr.max(1) - cellcorr.min(1) if scale_dtw else None
    dtcorr, route, dtw_score = ptalign_dtwscore(refpt, refcounts,
                                                cellpt, tumcounts,
                                                cdelta,
                                                n_corr_bins=n_corr_bins,
                                                min_dtw_bin_cells=min_dtw_bin_cells,
                                                scale_counts=scale_counts)

    if makeplots:
        plot_correlation_matrices(cellcorr, cellcorr_s, cellpt)
        if permutations == 0:
            plot_ptalign_metrics(cellpt, [], dtw_score, 1, dtcorr, route)

    # save outputs
    if 'all' in retain:
        retain = ['corrmat', 'scaled_corrmat', 'cellpt', 'dtw', 'route']

    if 'corrmat' in retain: ptalign['corrmat'] = cellcorr
    if 'scaled_corrmat' in retain: ptalign['scaled_corrmat'] = cellcorr_s
    if 'cellpt' in retain: ptalign['cellpt'] = cellpt
    if 'dtw' in retain: ptalign['dtw'] = dtcorr
    if 'route' in retain: ptalign['route'] = route

    # permute ptalign gene set and run predictions, returning DTW scores to compare
    if permutations > 0:
        print('step4: computing permuted ptalign')
        permres = permute_ptalign(raw_refcounts, refpt,
                                  raw_tumcounts[cellcorr.index], # excludes cells which fail cell_min_geneset_genes
                                  n_pt_bins, n_corr_bins, geneset,
                                  permutations=permutations,
                                  num_cores=num_cores,
                                  scale_counts=True,
                                  scale_dtw=True,
                                  cell_min_maxcorr=-1,
                                  retain=retain,
                                  keyargs=keyargs)

        # collect DTW metrics and compute permutation pvalue
        permscore = [permres[i]['dpscore'] for i in range(len(permres))]
        permpval = 1 - (len([i for i in permscore if i < dtw_score]) / permutations)

        if makeplots:
            plot_ptalign_metrics(cellpt, permscore, dtw_score, permpval, dtcorr, route)

        print('ptalign %s permutations: DTW-traceback p-value: %.2f' %(permutations, permpval))

        return ptalign, permres

    return ptalign
