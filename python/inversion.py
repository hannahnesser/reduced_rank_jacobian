import xarray as xr
import numpy as np
# from numpy import diag as diags
# from numpy import identity
from numpy.linalg import inv, norm, eigh
from scipy.sparse import diags, identity
from scipy.stats import linregress
# from scipy.linalg import eigh
import pandas as pd
from tqdm import tqdm
import copy

# clustering
from sklearn.cluster import KMeans

import math

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy

SCALE = 4
BASEFONT = 10
TITLE_LOC = 1.11
CBAR_PAD = 0.05
LABEL_PAD = 5
CBAR_LABEL_PAD = 75
rcParams['font.family'] = 'serif'
rcParams['font.size'] = BASEFONT*SCALE
rcParams['text.usetex'] = True

def color(k, cmap='inferno', lut=10):
    c = plt.cm.get_cmap(cmap, lut=lut)
    return colors.to_hex(c(k))

def cmap_trans(cmap, ncolors=300, nalpha=20):
    color_array = plt.get_cmap(cmap)(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.append(np.linspace(0.0,1.0,20), np.ones(ncolors-nalpha))

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='plasma_trans',colors=color_array)

    return map_object

class Inversion:
    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        print('... Initializing inversion object ...')
        # Check that the data are all the same types
        assert all(isinstance(z, np.ndarray)
                   for z in [k, xa, sa_vec, y, so_vec]), \
               'Input types aren\'t all numpy arrays.'

        # Define the state and observational dimensions
        self.nstate = k.shape[1]
        self.nobs = k.shape[0]
        self.latres = 1
        self.lonres = 1.25

        # Check whether all inputs have the right dimensions
        assert xa.shape[0] == self.nstate, \
               'Dimension mismatch: Jacobian and prior.'
        assert sa_vec.shape[0] == self.nstate, \
               'Dimension mismatch: Jacobian and prior error.'
        assert y.shape[0] == self.nobs, \
               'Dimension mismatch: Jacobian and observations.'
        assert so_vec.shape[0] == self.nobs, \
               'Dimension mismatch: Jacobian and observational error'

        # If everything works out, then we create the instance.
        self.k = k
        self.xa = xa
        self.sa_vec = sa_vec
        self.y = y
        self.y_base = y_base
        self.so_vec = so_vec

        # Force k to be positive
        if np.any(self.k < 0):
            print('Forcing negative values of the Jacobian to 0.')
            self.k[self.k < 0] = 0

        # Solve for the constant c.
        self.calculate_c()

        # Create space for a regularization factor.
        self.rf = 1

        # Now create some holding spaces for values that may be filled
        # in the course of solving the inversion.
        self.xhat = None
        self.shat = None
        self.a = None
        self.y_out = None

        print('... Complete ...\n')


    ####################################
    ### STANDARD INVERSION FUNCTIONS ###
    ####################################

    def calculate_c(self):
        self.c = self.y_base - self.k @ self.xa

    def obs_mod_diff(self, x):
        return self.y - (self.k @ x + self.c)

    def cost_func(self, x):
        cost_obs = self.obs_mod_diff(x).T \
                   @ diags(self.rf/self.so_vec) @ self.obs_mod_diff(x)
        cost_emi = (x - self.xa).T @ diags(1/self.sa_vec) @ (x - self.xa)
        print('     Cost function: %.2f' % (cost_obs + cost_emi))
        return cost_obs + cost_emi

    def solve_inversion(self):
        print('... Solving inversion ...')
        so_inv = diags(self.rf/self.so_vec)
        sa_inv = diags(1/self.sa_vec)

        # Calculate the cost function at the prior.
        print('Calculating the cost function at the prior mean.')
        cost_prior = self.cost_func(self.xa)

        # Calculate the posterior error.
        print('Calculating the posterior error.')
        self.shat = np.asarray(inv(self.k.T @ so_inv @ self.k + sa_inv))

        # Calculate the posterior mean
        print('Calculating the posterior mean.')
        gain = np.asarray(self.shat @ self.k.T @ so_inv)
        self.xhat = self.xa + (gain @ self.obs_mod_diff(self.xa))

        # Calculate the cost function at the posterior. Also
        # calculate the number of negative cells as an indicator of
        # inversion success.
        print('Calculating the cost function at the posterior mean.')
        cost_post = self.cost_func(self.xhat)
        print('     Negative cells: %d' % self.xhat[self.xhat < 0].sum())

        # Calculate the averaging kernel.
        print('Calculating the averaging kernel.')
        self.a = np.asarray(identity(self.nstate) \
                            - self.shat @ sa_inv)
        # self.dofs = np.diag(self.a)
        print('     DOFS: %.2f' % np.trace(self.a))

        # Calculate the new set of modeled observations.
        print('Calculating updated modeled observations.')
        self.y_out = self.k @ self.xhat + self.c

        print('... Complete ...\n')

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################

    @staticmethod
    def match_data_to_clusters(data, clusters, default_value=0):
        result = clusters.copy()
        c_array = result.values
        c_idx = np.where(c_array > 0)
        c_val = c_array[c_idx]
        row_idx = [r for _, r, _ in sorted(zip(c_val, c_idx[0], c_idx[1]))]
        col_idx = [c for _, _, c in sorted(zip(c_val, c_idx[0], c_idx[1]))]
        idx = (row_idx, col_idx)

        d_idx = np.where(c_array == 0)

        c_array[c_idx] = data
        c_array[d_idx] = default_value
        result.values = c_array

        return result

    @staticmethod
    def plot_state_format(data, default_value=0, cbar=True, **kw):
        # Get kw
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(8*SCALE/1.25,6*SCALE/1.25),
                                   subplot_kw={'projection' :
                                               ccrs.PlateCarree()})

        title = kw.pop('title', '')
        kw['cmap'] = kw.get('cmap', 'viridis')
        kw['vmin'] = kw.get('vmin', data.min())
        kw['vmax'] = kw.get('vmax', data.max())
        kw['add_colorbar'] = False

        c = data.plot(ax=ax, snap=True, **kw)

        ax.set_title(title, y=TITLE_LOC, fontsize=(BASEFONT+10)*SCALE)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98')
        ax.add_feature(cartopy.feature.LAND, facecolor='0.98')
        ax.coastlines(color='grey')
        gl = ax.gridlines(linestyle=':', draw_labels=True, color='grey')
        gl.xlabel_style = {'fontsize' : (BASEFONT-5)*SCALE}
        gl.ylabel_style = {'fontsize' : (BASEFONT-5)*SCALE}

        if cbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.05,
                                ax.get_position().y0,
                                0.005*SCALE,
                                ax.get_position().height])
            cb = fig.colorbar(c, ax=ax, cax=cax)
            cb.ax.tick_params(labelsize=BASEFONT*SCALE)
            return fig, ax, cb
        else:
            return fig, ax, c

    def plot_state(self, attribute, clusters_plot, default_value=0,
                   cbar=True, **kw):
        # Get the data from the attribute argument. The attribute argument
        # is either a string or a string and int. ## EXPAND COMMENT HERE
        try:
            attribute, selection = attribute
            data = getattr(self, attribute)[:, selection]
            attribute_str = attribute + '_' + str(selection)
        except ValueError:
            data = getattr(self, attribute)
            attribute_str = attribute

        kw['title'] = kw.get('title', attribute_str)

        # Force the data to have dimension equal to the state dimension
        assert data.shape[0] == self.nstate, \
               'Dimension mismatch: Data does not match state dimension.'

        # Match the data to lat/lon data
        data = self.match_data_to_clusters(data, clusters_plot, default_value)

        # Plot
        fig, ax, c = self.plot_state_format(data, default_value, cbar, **kw)
        return fig, ax, c

    def plot_grid(self, attributes, nx, ny, clusters_plot, **kw):
        assert nx*ny == len(attributes), \
               'Dimension mismatch: Data does not match number of plots.'

        try:
            kw.get('vmin')
            kw.get('vmax')
        except KeyError:
            print('vmin and vmax not supplied. Plots may have inconsistent\
                   colorbars.')

        try:
            titles = kw.pop('titles')
            vmins = kw.pop('vmins')
            vmaxs = kw.pop('vmaxs')
        except KeyError:
            pass

        fig, ax = plt.subplots(nx, ny, figsize=(ny*2*5.25*SCALE/2, nx*6.75*SCALE/2),
                               subplot_kw={'projection' : ccrs.PlateCarree()})
        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        # cax = fig.add_axes([0.95, 0.25/2, 0.01, 0.75])
        cax = fig.add_axes([ax.get_position().x1 + 0.05,
                            ax.get_position().y0,
                            0.005*SCALE,
                            ax.get_position().height])

        # kw['add_colorbar'] = False
        cbar_kwargs = kw.pop('cbar_kwargs', {})

        for i, axis in enumerate(ax.flatten()):
            kw['figax'] = [fig, axis]
            try:
                kw['title'] = titles[i]
                kw['vmin'] = vmins[i]
                kw['vmax'] = vmaxs[i]
            except NameError:
                pass
            fig, axis, c = self.plot_state(attributes[i], clusters_plot,
                                           cbar=False, **kw)

        fig.colorbar(c, cax=cax, **cbar_kwargs)
        cax.tick_params(labelsize=BASEFONT*SCALE)

        return fig, ax, cax


class ReducedRankInversion(Inversion):
    # class variables shared by all instances

    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        # We inherit from the inversion class and create space for
        # the reduced rank solutions.
        Inversion.__init__(self, k, xa, sa_vec, y, y_base, so_vec)

        # We create space for the rank
        self.rank = None

        # We also want to save the eigendecomposition values
        self.evals_q = None
        self.evals_h = None
        self.evecs = None

        # and for the final solutions
        self.xhat_proj = None
        self.xhat_kproj = None
        self.xhat_fr = None

        self.shat_proj = None
        self.shat_kproj = None

        self.a_proj = None
        self.a_kproj = None

    ########################################
    ### REDUCED RANK INVERSION FUNCTIONS ###
    ########################################

    def get_rank(self, pct_of_info=None, rank=None, snr=None):
        if sum(x is not None for x in [pct_of_info, rank, snr]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank, snr]) == 0:
            raise AttributeError('Must provide one of pct_of_info, rank, or snr.')
        elif pct_of_info is not None:
            frac = np.cumsum(self.evals_q/self.evals_q.sum())
            diff = np.abs(frac - pct_of_info)
            rank = np.argwhere(diff == np.min(diff))[0][0]
            print('Calculated rank from percent of information: %d' % rank)
        elif snr is not None:
            diff = np.abs(self.evals_h - snr)
            rank = np.argwhere(diff == np.min(diff))[0][0]
            print('Calculated rank from signal-to-noise ratio : %d' % rank)
        elif rank is not None:
            print('Using defined rank: %d' % rank)
        return rank

    def pph(self):
        # Calculate the prior pre-conditioned Hessian
        sa_sqrt = diags(self.sa_vec**0.5)
        so_inv = diags(self.rf/self.so_vec)
        pph_m  = sa_sqrt @ self.k.T \
                 @ so_inv @ self.k @ sa_sqrt
        print('Calculated PPH.')
        return pph_m

    def edecomp(self):
        print('... Calculating eigendecomposition ...')
        # Calculate pph and require that it be symmetric
        pph = self.pph()
        assert np.allclose(pph, pph.T, rtol=1e-5), \
               'The prior pre-conditioned Hessian is not symmetric.'

        # Perform the eigendecomposition of the prior
        # pre-conditioned Hessian
        # We return the evals of the projection, not of the
        # prior pre-conditioned Hessian.

        evals, evecs = eigh(pph)
        print('Eigendecomposition complete.')

        # Sort evals and evecs by eval
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:,idx]

        # Force all evals to be non-negative
        if (evals < 0).sum() > 0:
            print('Negative eigenvalues. Maximum negative value is %.2e. Setting negative eigenvalues to zero.' \
                % (evals[evals < 0].min()))
            evals[evals < 0] = 0

        # Check for imaginary eigenvector components and force all
        # eigenvectors to be only the real component.
        if np.any(np.iscomplex(evecs)):
            print('Imaginary eigenvectors exist at index %d of %d. Forcing eigenvectors to real component alone.' \
                  % ((np.where(np.iscomplex(evecs))[1][0] - 1), len(evecs)))
            evecs = np.real(evecs)

        # Saving result to our instance.
        print('Saving eigenvalues and eigenvectors to instance.')
        # self.evals = evals/(1 + evals)
        self.evals_h = evals
        self.evals_q = evals/(1 + evals)
        self.evecs = evecs
        print('... Complete ...\n')

    def projection(self, pct_of_info=None, rank=None, snr=None):
        # Conduct the eigendecomposition of the prior pre-conditioned
        # Hessian
        if ((self.evals_h is None) or
            (self.evals_q is None) or
            (self.evecs is None)):
            self.edecomp()

        # Subset the evecs according to the rank provided.
        rank = self.get_rank(pct_of_info=pct_of_info, rank=rank, snr=snr)
        evecs_subset = self.evecs[:,:rank]

        # Calculate the prolongation and reduction operators and
        # the resulting projection operator.
        prolongation = (evecs_subset.T * self.sa_vec**0.5).T
        reduction = (1/self.sa_vec**0.5) * evecs_subset.T
        projection = prolongation @ reduction

        return rank, prolongation, reduction, projection

    # def shat_proj_sum(self, rank):
    #     sum_mat = np.zeros((self.nstate, self.nstate))
    #     for i in range(rank):
    #         l_i = self.evals[i]
    #         v_i = self.evecs[:,i].reshape(-1,1)
    #         sum_mat += (v_i @ v_i.T)/(1 + l_i)
    #     return sum_mat

    # Need to add in cost function and other information here
    def solve_inversion_proj(self, pct_of_info=None, rank=None):
        print('... Solving projected inversion ...')
        # Conduct the eigendecomposition of the prior pre-conditioned
        # Hessian
        if ((self.evals_h is None) or
            (self.evals_q is None) or
            (self.evecs is None)):
            self.edecomp()

        # Subset the evecs according to the rank provided.
        rank = self.get_rank(pct_of_info=pct_of_info, rank=rank)

        # Calculate a few quantities that will be useful
        sa_sqrt = diags(self.sa_vec**0.5)
        sa_sqrt_inv = diags(1/self.sa_vec**0.5)
        # so_vec = self.rf*self.so_vec
        so_inv = diags(self.rf/self.so_vec)
        # so_sqrt_inv = diags(1/so_vec**0.5)

        # Subset evecs and evals
        vk = self.evecs[:, :rank]
        # wk = so_sqrt_inv @ self.k @ sa_sqrt @ vk
        lk = self.evals_h[:rank].reshape((1, -1))

        # Make lk into a matrix
        lk = np.repeat(lk, self.nstate, axis=0)

        # Calculate the solutions
        self.xhat_proj = (np.asarray(sa_sqrt
                                    @ ((vk/(1+lk)) @ vk.T)
                                    @ sa_sqrt @ self.k.T @ so_inv
                                    @ self.obs_mod_diff(self.xa))
                         + self.xa)
        self.shat_proj = np.asarray(sa_sqrt
                                    @ (((1/(1+lk))*vk) @ vk.T)
                                    @ sa_sqrt)
        # self.shat_proj = sa_sqrt @ self.shat_proj_sum(rank) @ sa_sqrt
        self.a_proj = np.asarray(sa_sqrt
                                 @ (((lk/(1+lk))*vk) @ vk.T)
                                 @ sa_sqrt_inv)
        print('... Complete ...\n')

    def solve_inversion_kproj(self, pct_of_info=None, rank=None):
        print('... Solving projected Jacobian inversion ...')
        # Get the projected solution
        self.solve_inversion_proj(pct_of_info=pct_of_info, rank=rank)

        # Calculate a few quantities that will be useful
        sa = diags(self.sa_vec)

        # Calculate the solutions
        self.xhat_kproj = self.xhat_proj
        self.shat_kproj = np.asarray((identity(self.nstate) - self.a_proj) @ sa)
        self.a_kproj = self.a_proj
        print('... Complete ...\n')

    def solve_inversion_fr(self, pct_of_info=None, rank=None):
        print('... Solving full rank approximation inversion ...')
        self.solve_inversion_kproj(pct_of_info=pct_of_info, rank=rank)

        so_inv = diags(self.rf/self.so_vec)
        d = self.obs_mod_diff(self.xa)

        self.xhat_fr = self.shat_kproj @ self.k.T @ so_inv @ d
        print('... Complete ...\n')


    ##########################
    ########## ERROR #########
    ##########################

    def calc_error(self, attribute, compare_data):
        '''
        self = truth  (x axis)
        compare_data = y axis
        '''
        e = compare_data
        t = getattr(self, attribute)
        err_abs = np.linalg.norm(t - e)
        err_rel = err_abs/np.linalg.norm(t)
        return err_abs, err_rel

    ##########################
    ### PLOTTING FUNCTIONS ###
    ##########################

    def plot_eval_spectra(self, **kw):
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(7*SCALE/1.25, 5*SCALE/1.25))
        label = kw.pop('label', '')
        color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
        ls = kw.pop('ls', '-')
        if kw:
            raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

        ax.plot(self.evals_q, label=label, c=color, ls=ls)

        ax.set_facecolor('0.98')
        ax.legend(frameon=False, fontsize=(BASEFONT+5)*SCALE)
        ax.set_xlabel('Eigenvalue Index', fontsize=(BASEFONT+5)*SCALE)
        ax.set_ylabel(r'Q$_{DOF}$ Eigenvalues', fontsize=(BASEFONT+5)*SCALE)
        plt.tick_params(axis='both', which='major', labelsize=BASEFONT*SCALE)

        return fig, ax

    def plot_info_frac(self, **kw):
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(7*SCALE/1.25, 5*SCALE/1.25))
        label = kw.pop('label', '')
        color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
        ls = kw.pop('ls', '-')
        lw = kw.pop('lw', 3)
        text = kw.pop('text', True)
        if kw:
            raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

        frac = np.cumsum(self.evals_q/self.evals_q.sum())
        snr_idx = np.argwhere(self.evals_q >= 0.5)[-1][0]
        ax.plot(frac, label=label, c=color, ls=ls, lw=lw)

        if text:
            ax.scatter(snr_idx, frac[snr_idx], s=10*SCALE, c=color)
            ax.text(snr_idx + self.nstate*0.01, frac[snr_idx],
                    r'SNR $\approx$ 1',
                    ha='left', va='top', fontsize=BASEFONT*SCALE,
                    color=color)
            ax.text(snr_idx + self.nstate*0.01, frac[snr_idx] - 0.1,
                    'n = %d' % snr_idx,
                    ha='left', va='top', fontsize=BASEFONT*SCALE,
                    color=color)
            ax.text(snr_idx + self.nstate*0.01, frac[snr_idx] - 0.2,
                    r'$f_{DOFS}$ = %.2f' % frac[snr_idx],
                    ha='left', va='top', fontsize=BASEFONT*SCALE,
                    color=color)

        ax.set_facecolor('0.98')
        ax.legend(frameon=False, fontsize=(BASEFONT+5)*SCALE)
        ax.set_xlabel('Eigenvector Index', fontsize=(BASEFONT+5)*SCALE)
        ax.set_ylabel('Fraction of DOFS', fontsize=(BASEFONT+5)*SCALE)
        plt.tick_params(axis='both', which='major', labelsize=BASEFONT*SCALE)

        return fig, ax


    # def plot_evec_comparison(self, compare_data, **kw):
    #     try:
    #         fig, ax = kw.pop('figax')
    #     except KeyError:
    #         fig, ax = plt.subplots(figsize=(10, 5))
    #     label = kw.pop('label', '')
    #     color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
    #     ls = kw.pop('ls', '-')
    #     if kw:
    #         raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

    #     diff_neg = np.linalg.norm(self.evecs - compare_data.evecs, axis=0)
    #     diff_pos = np.linalg.norm(self.evecs + compare_data.evecs, axis=0)
    #     rel_diff = np.minimum(diff_neg, diff_pos)/np.linalg.norm(self.evecs, axis=0)
    #     ax.plot(rel_diff, label=label, c=color, ls=ls)

    #     ax.set_facecolor('0.98')
    #     ax.set_xlabel('Eigenvalue Index', fontsize=22)
    #     ax.set_ylabel(r'Eigenvector Difference', fontsize=22)
    #     ax.set_ylim(0,3)
    #     plt.tick_params(axis='both', which='major', labelsize=18)

    #     return fig, ax

    @staticmethod
    def calc_stats(xdata, ydata):
        m, b, r, p, err = linregress(xdata.flatten(),
                                     ydata.flatten())
        return m, b, r

    def plot_comparison(self, attribute, compare_data,
                        cbar=True,
                        stats=True, **kw):
        # get attribute data
        xdata = getattr(self, attribute)

        # Get plotting kwargs
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(7*SCALE/1.25, 7*SCALE/1.25))

        xlabel = kw.pop('xlabel', 'Truth')
        ylabel = kw.pop('ylabel', 'Estimate')
        title = kw.pop('title', 'Estimated vs. True ' + attribute)

        if type(compare_data) == dict:
           # We need to know how many data sets were passed
            n = len(compare_data)
            cmap = kw.pop('cmap', 'inferno')
            cbar_ticklabels = kw.pop('cbar_ticklabels',
                                     list(compare_data.keys()))
            cbar_title = kw.pop('cbar_title', '')

            # Plot data
            count = 0
            for k, ydata in compare_data.items():
                ax.scatter(xdata, ydata,
                           alpha=0.5, s=5*SCALE, c=color(count, cmap=cmap, lut=n))
                count += 1

            # Color bar
            # cax = fig.add_axes([0.95, 0.25/2, 0.01, 0.75])
            cax = fig.add_axes([ax.get_position().x1 + 0.05,
                                ax.get_position().y0,
                                0.005*SCALE,
                                ax.get_position().height])
            norm = colors.Normalize(vmin=0, vmax=n)
            cb = colorbar.ColorbarBase(cax,
                                       cmap=plt.cm.get_cmap(cmap, lut=n),
                                       norm=norm)
            cb.set_ticks(0.5 + np.arange(0,n+1))
            cb.set_ticklabels(cbar_ticklabels)
            cb.set_label(cbar_title, fontsize=(BASEFONT+5)*SCALE)
            plt.tick_params(axis='both', which='both', labelsize=BASEFONT*SCALE)

        else:
            c = kw.pop('color',
                       np.asarray(plt.cm.get_cmap('inferno',
                                                  lut=10)(3)).reshape(1,-1))
            # ax.scatter(xdata, compare_data,
            #            alpha=0.5, s=5*SCALE, c=c)

            # Replace with hexbin

            # Get data limits
            dmin = min(np.min(xdata), np.min(compare_data))
            dmax = max(np.max(xdata), np.max(compare_data))
            pad = (dmax - dmin)*0.05
            dmin -= pad
            dmax += pad

            try:
                # get lims
                ylim = kw.pop('ylim')
                xlim = kw.pop('xlim')
                xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
            except:
                # set lims
                xlim = ylim = xy = (dmin, dmax)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            bin_max = len(self.xhat)/10
            round_by = len(str(len(self.xhat)/20).split('.')[0]) - 1
            bins = np.arange(0, int(round(bin_max, -round_by)))
            gridsize = math.floor((dmax - dmin)/(xy[1] - xy[0])*40)
            c = ax.hexbin(xdata, compare_data,
                          cmap=cmap_trans('plasma_r'),
                          bins=bins,
                          gridsize=gridsize)

            if stats:
                # Error
                # err_abs, err_rel = self.calc_error(attribute, compare_data)
                m, b, r = self.calc_stats(xdata, compare_data)
                xs = np.array([dmin, dmax])
                ys = m*xs + b
                # ax.plot(xs, ys, c=color(6))
                if r**2 <= 0.99:
                    ax.text(0.05, 0.9,
                            r'R$^2$ = %.2f' % r**2,
                            fontsize=(BASEFONT+5)*SCALE,
                            transform=ax.transAxes)
                else:
                    ax.text(0.05, 0.9,
                            r'R$^2$ $>$ 0.99',
                            fontsize=(BASEFONT+5)*SCALE,
                            transform=ax.transAxes)
                # ax.text(0.05, 0.825,
                #         r'Slope = %.2f' % m,
                #         fontsize=(BASEFONT+5)*SCALE,
                #         transform=ax.transAxes)
                # ax.text(0.05, 0.75,
                #         r'Intercept = %.2f' % b,
                #         fontsize=(BASEFONT+5)*SCALE,
                #         transform=ax.transAxes)

        # Aesthetics
        ax.set_facecolor('0.98')

        ax.plot(xy, xy, c='0.1', lw=2, ls=':', alpha=0.5, zorder=0)
        ax.set_xlabel(xlabel, fontsize=(BASEFONT+6)*SCALE,
                      labelpad=LABEL_PAD)
        ax.set_ylabel(ylabel, fontsize=(BASEFONT+6)*SCALE,
                      labelpad=LABEL_PAD)
        ax.set_title(title, fontsize=(BASEFONT+10)*SCALE, y=TITLE_LOC-0.06)
        ax.tick_params(axis='both', which='both', labelsize=BASEFONT*SCALE)

        if cbar:
            # cax = fig.add_axes([0.925, 0.25/2, 0.015, 0.75])
            cax = fig.add_axes([ax.get_position().x1 + 0.05,
                                ax.get_position().y0,
                                0.005*SCALE,
                                ax.get_position().height])
            cb = plt.colorbar(c, cax=cax, boundaries=bins)
            cb.ax.tick_params(labelsize=BASEFONT*SCALE)
            return fig, ax, cb
        else:
            return fig, ax, c

    # def calc_difference(self, attribute, compare_data, norm_func=norm):


class ReducedRankJacobian(ReducedRankInversion):
    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        # Inherit from the parent class.
        ReducedRankInversion.__init__(self, k, xa, sa_vec, y, y_base, so_vec)

        self.perturbed_cells = np.array([])
        self.model_runs = 0


    #######################################
    ### REDUCED RANK JACOBIAN FUNCTIONS ###
    #######################################
    # def create_aggregate_cells_kmeans():

    def get_neighboring_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        neighboring_cells = clusters_plot.where((clusters_plot.lon <= ulim_lon) &
                                                (clusters_plot.lon >= llim_lon) &
                                                (clusters_plot.lat <= ulim_lat) &
                                                (clusters_plot.lat >= llim_lat),
                                                drop=True).values.flatten()
        return neighboring_cells[neighboring_cells > 0].astype(int)

    def get_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        cond_lon = (clusters_plot.lon <= ulim_lon) & \
                   (clusters_plot.lon >= llim_lon) & \
                   (clusters_plot.lat == latlon.lat.values)
        cond_lat = (clusters_plot.lat <= ulim_lat) & \
                   (clusters_plot.lat >= llim_lat) & \
                   (clusters_plot.lon == latlon.lon.values)
        cond = xr.ufuncs.logical_or(cond_lon, cond_lat)
        adjacent_cells = clusters_plot.where(cond,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]

        # get rid of kitty corner nans
        adjacent_cells = adjacent_cells[~np.isnan(adjacent_cells)]

        # get rid of 0s
        adjacent_cells = adjacent_cells[adjacent_cells > 0]

        return adjacent_cells.astype(int)

    def get_lat_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lat = latlon.lat.values + ring*self.latres
        llim_lat = latlon.lat.values - ring*self.latres
        cond_lat = (clusters_plot.lat <= ulim_lat) & \
                   (clusters_plot.lat >= llim_lat) & \
                   (clusters_plot.lon == latlon.lon.values)
        adjacent_cells = clusters_plot.where(cond_lat,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]
        return adjacent_cells[~np.isnan(adjacent_cells)].astype(int)

    def get_lon_adjacent_cells(self, nsv_index, clusters_plot, ring=1):
        latlon = clusters_plot.where(clusters_plot == nsv_index, drop=True)
        ulim_lon = latlon.lon.values + ring*self.lonres
        llim_lon = latlon.lon.values - ring*self.lonres
        cond_lon = (clusters_plot.lon <= ulim_lon) & \
                   (clusters_plot.lon >= llim_lon) & \
                   (clusters_plot.lat == latlon.lat.values)
        adjacent_cells = clusters_plot.where(cond_lon,
                                             drop=True).values.flatten()
        adjacent_cells = adjacent_cells[adjacent_cells != nsv_index]
        return adjacent_cells[~np.isnan(adjacent_cells)].astype(int)

    def merge_cells_kmeans(self, label_idx, clusters_plot, cluster_size):
        labels = np.zeros(self.nstate)
        labels[label_idx] = label_idx
        labels = self.match_data_to_clusters(labels,
                                               clusters_plot)
        labels = labels.to_dataframe('labels').reset_index()
        labels = labels[labels['labels'] > 0]

        # Now do kmeans clustering
        n_clusters = int(len(label_idx)/cluster_size)
        labels_new = KMeans(n_clusters=n_clusters).fit(labels[['lat',
                                                              'lon']])

        # Print out some information
        label_stats = np.unique(labels_new.labels_, return_counts=True)
        print('Number of clusters: %d' % len(label_stats[0]))
        print('Cluster size: %d' % cluster_size)
        print('Maximum number of grid boxes in a cluster: %d' \
              % max(label_stats[1]))

        # save the information
        labels = labels.assign(new_labels=labels_new.labels_+1)
        labels[['labels', 'new_labels']] = labels[['labels',
                                                   'new_labels']].astype(int)

        return labels

    def aggregate_cells_kmeans(self, clusters_plot, significance,
                               n_cells, n_cluster_size=None):
        # Create a new vector that will contain the updated state
        # vector indices, with 0s elsewhere
        new_sv = np.zeros(self.nstate)

        # Get the indices associated with the most significant
        # grid boxes
        # sig_idx = significance.argsort()[::-1]
        sig_idx = self.dofs.argsort()[::-1]

        # Iterate through n_cells
        n_cells = np.append(0, n_cells)
        nidx = np.cumsum(n_cells)
        for i, n in enumerate(n_cells[1:]):
            # get cluster size
            if n_cluster_size is None:
                cluster_size = i+1
            else:
                cluster_size = n_cluster_size[i]

            # get the indices of interest
            sub_sig_idx = sig_idx[nidx[i]:nidx[i+1]]

            new_labels = self.merge_cells_kmeans(sub_sig_idx,
                                                 clusters_plot,
                                                 cluster_size)

            new_sv[new_labels['labels']] = new_labels['new_labels']+new_sv.max()

        print('Number of state vector elements: %d' \
              % len(np.unique(new_sv)[1:]))
        return new_sv

    # @staticmethod
    # def check_similarity(old_sv, new_sv):


    @staticmethod
    def calculate_perturbation_matrix(state_vector, significance):
        sv_sig = pd.DataFrame({'sv' : state_vector,
                               'sig' : significance})
        sv_sig = sv_sig.groupby(by='sv').mean().reset_index()
        sv_sig = sv_sig.sort_values(by='sig')
        sv_sig = sv_sig[sv_sig['sv'] > 0]
        p = np.zeros((len(state_vector), sv_sig.shape[0]))
        for i, sv in enumerate(sv_sig['sv']):
            index = np.argwhere(state_vector == sv).reshape(-1,)
            p[index, i] = 0.5
        return p

    def calculate_k(self, forward_model, state_vector, k_base):
        sv_elements = np.unique(state_vector)[1:]
        for i in sv_elements:
            index = np.argwhere(state_vector == i).reshape(-1,)
            dx = np.zeros(self.nstate)
            dx[index] = 0.5
            dy = forward_model @ dx
            ki = dy/(0.5*len(index))
            ki = np.tile(ki.reshape(-1,1), (1,len(index)))
            k_base[:,index] = ki
        return k_base

    def calculate_significance(self,
                               pct_of_info=None, rank=None, prolongation=None):
        if prolongation is None:
            rank, prolongation, _, _ = self.projection(pct_of_info=pct_of_info,
                                                       rank=rank)
        significance = np.sqrt((prolongation**2)).sum(axis=1)

        return rank, significance

    def broyden(self, forward_model, perturbation_matrix, factor=10):
        # perturbation_diff = np.diff(perturbation_matrix, axis=1)
        # for p in perturbation_diff.T:
        #     p = p.reshape(-1, 1)
        #     self.k += ((forward_model - self.k) @ p @ p.T)/(p**2).sum()
        for i in range(perturbation_matrix.shape[1] - factor):
            p = perturbation_matrix[:, i:(i + factor)]
            p = p.reshape(-1, factor)
            self.k += ((forward_model - self.k) @ p @ p.T)/(p**2).sum()

    def update_jacobian_ag(self, forward_model, clusters_plot,
                        pct_of_info=None,
                        n_cells=[100, 200],
                        n_cluster_size=[1, 2],
                        k_base=None):#, #threshold=0,
                        # perturbation_factor=0.5):

        if k_base is None:
            k_base = copy.deepcopy(self.k)

        # We start by creating a new instance of the class. This
        # is where
        new = ReducedRankJacobian(k=k_base,
                                  xa=copy.deepcopy(self.xa),
                                  sa_vec=copy.deepcopy(self.sa_vec),
                                  y=copy.deepcopy(self.y),
                                  y_base=copy.deepcopy(self.y_base),
                                  so_vec=copy.deepcopy(self.so_vec))
        new.model_runs = copy.deepcopy(self.model_runs)
        new.rf = copy.deepcopy(self.rf)

        # # Retrieve the prolongation operator associated with
        # # this instance of the Jacobian for the rank specified
        # # by the function call. These are the eigenvectors
        # # that we will perturb.
        # # Calculate the weight of each full element space as sum
        # # of squares
        # new.rank, significance = self.calculate_significance(pct_of_info)

        # Replace with dofs
        self.dofs = np.trace(self.a)

        # If previously optimized, set significance to 0
        if len(self.perturbed_cells) > 0:
            print('Ignoring previously optimized grid cells.')
            # significance[self.perturbed_cells] = 0
            self.dofs[self.perturbed_cells] = 0

        # We need the new state vector first. This gives us the
        # clusterings of the base resolution state vector
        # elements as dictated by n_cells and n_cluster_size.
        state_vector = self.aggregate_cells_kmeans(clusters_plot,
                                                   n_cells,
                                                   n_cluster_size)

        #### CHECK ######
        # And now get the perturbation matrix
        perturbation_matrix = self.calculate_perturbation_matrix(state_vector,
                                                                 significance)

        # We calculate the number of model runs.
        new.model_runs += len(np.unique(state_vector)[1:])

        # Find the individual grid cells that are perturbed
        counts = np.unique(state_vector, return_counts=True)
        new_perturbed_cells = counts[0][counts[1] == 1]
        new.perturbed_cells = np.append(self.perturbed_cells,
                                        new_perturbed_cells).astype(int)

        # # Now update the Jacobian
        # k_new = self.calculate_k(forward_model, state_vector, k_new)
        # new.k = k_new
        new.broyden(forward_model,
                    perturbation_matrix=perturbation_matrix)

        # Update the value of c in the new instance
        new.calculate_c()

        # And do the eigendecomposition
        new.edecomp()

        # And solve the inversion
        new.solve_inversion()

        return new

    # Rewrite to take any prolongation matrix?
    def update_jacobian(self, forward_model,
                        pct_of_info=None, rank=None, snr=None,
                        prolongation=None, reduction=None):#,
                        # k_base=None,
                        # convergence_threshold=1e-4, max_iterations=1000):
        # We start by creating a new instance of the class.
        # if k_base is None:
        #     k_base = copy.deepcopy(self.k)

        # Retrieve the prolongation operator associated with
        # this instance of the Jacobian for the rank specified
        # by the function call. These are the eigenvectors
        # that we will perturb.
        # if (pct_of_info is not None) and (rank is not None):
        if sum(x is not None for x in [pct_of_info, rank, snr, prolongation]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank, snr, prolongation]) == 0:
            raise AttributeError('Must provide one of pct_of_info, rank, or prolongation.')
        elif (((prolongation is not None) and (reduction is None)) or
              ((prolongation is None) and (reduction is not None))):
            raise AttributeError('Only one of prolongation or reduction is provided.')
        elif (prolongation is not None) and (reduction is not None):
            print('Using given prolongation and reduction.')
        else:
            print('Calculating prolongation and reduction.')
            _, prolongation, reduction, _ = self.projection(pct_of_info=pct_of_info,
                                                            rank=rank,
                                                            snr=snr)

        # Run the perturbation runs
        perturbations = forward_model @ prolongation

        # Project to the full dimension
        k = perturbations @ reduction

        # Save to a new instance
        new = ReducedRankJacobian(k=k,
                                  xa=copy.deepcopy(self.xa),
                                  sa_vec=copy.deepcopy(self.sa_vec),
                                  y=copy.deepcopy(self.y),
                                  y_base=copy.deepcopy(self.y_base),
                                  so_vec=copy.deepcopy(self.so_vec))
        new.model_runs = copy.deepcopy(self.model_runs) + prolongation.shape[1]
        new.rf = copy.deepcopy(self.rf)

        # Do the eigendecomposition
        new.edecomp()

        # And solve the inversion
        new.solve_inversion()
        # new.solve_inversion_kproj(rank=floor(rank/2))

        return new

    def filter(self, true, mask):
        self_f = copy.deepcopy(self)
        true_f = copy.deepcopy(true)

        skeys = [k for k in dir(self_f) if k[:4] == 'shat']
        xkeys = [k for k in dir(self_f) if k[:4] == 'xhat']
        akeys = [k for k in dir(self_f) if (k == 'a') or (k[:2] == 'a_')]

        for obj in [true_f, self_f]:
            # update jacobian
            setattr(obj, 'k', getattr(obj, 'k')[:, mask])
            for keylist in [skeys, xkeys, akeys]:
                for k in keylist:
                    try:
                        if getattr(obj, k).ndim == 1:
                            # update true_f and self_f posterior mean
                            setattr(obj, k, getattr(obj, k)[mask])
                        else:
                            # update true_f and self_f posterior variance
                            setattr(obj, k, getattr(obj, k)[mask, :][:, mask])
                    except AttributeError:
                        pass

        # some plotting functions
        true_f.xhat_long = copy.deepcopy(true.xhat)
        true_f.xhat_long[~mask] = 1
        self_f.xhat_long = copy.deepcopy(self.xhat)
        self_f.xhat_long[~mask] = 1

        return self_f, true_f

    def full_analysis(self, true, clusters_plot):
        if ((self.xhat_fr is None) or (true.xhat_fr is None)):
            print('Reduced rank inversion is not solved.')
            fig1, ax = plt.subplots(1, 4, figsize=(22*SCALE, 4.5*SCALE))
            axis = ax
        else:
            fig1, ax = plt.subplots(2, 4, figsize=(16*4/3*SCALE, 11*SCALE))

            # Projected estimated posterior vs. true posterior
            title = r'$\tilde{\hat{x}_{K\Pi}}\ vs.\ \hat{x}$'
            fig1, ax[1, 0], c = true.plot_comparison('xhat_kproj', self.xhat,
                                                     cbar=False,
                                                     **{'figax' : [fig1, ax[1, 0]],
                                                        'title' : title,
                                                        'xlabel' : 'Truth',
                                                        'ylabel' : 'Estimate'})

            # Full rank posterior vs. true posterior
            title = r'$\tilde{\hat{x}_{FR}}\ vs.\ \hat{x}$'
            fig1, ax[1, 1], c = true.plot_comparison('xhat_fr', self.xhat,
                                                     cbar=False,
                                                     **{'figax' : [fig1, ax[1, 1]],
                                                        'title' : title,
                                                        'xlabel' : 'Truth',
                                                        'ylabel' : 'Estimate'})

            # Projected error vs. true error
            true.shat_diag = np.diag(true.shat)
            self.shat_kproj_diag = np.diag(self.shat_kproj)
            title = r'$\tilde{\hat{S}_{K\Pi}}$ vs. $\hat{S}$'
            fig1, ax[1, 2], c = true.plot_comparison('shat_iag', self.shat_kproj_diag,
                                                     cbar=False,
                                                      **{'figax' : [fig1, ax[1, 2]],
                                                         'title' : title,
                                                         'xlabel' : 'Truth',
                                                         'ylabel' : 'Estimate'})
            del true.shat_diag
            del self.shat_kproj_diag


            title = r'$\tilde{A_{K\Pi}}$ vs. $A$'
            true.a_diag = np.diag(true.a)
            self.a_kproj_diag = np.diag(self.a_kproj)
            fig1, ax[1, 3], c = true.plot_comparison('a_diag', self.a_kproj_diag,
                                                     cbar=False,
                                                      **{'figax' : [fig1, ax[1, 2]],
                                                         'title' : title,
                                                         'xlabel' : 'Truth',
                                                         'ylabel' : 'Estimate'})

            axis = ax[0, :]

        # Full dimension Jacobian
        fig1, axis[0], c = true.plot_comparison('k', self.k,
                                                cbar=False,
                                                **{'figax' : [fig1, axis[0]],
                                                   'title' : r'Jacobian',
                                                   'xlabel' : 'Truth',
                                                   'ylabel' : 'Estimate'})

        # Posterior
        fig1, axis[1], c = true.plot_comparison('xhat', self.xhat,
                                                cbar=False,
                                                **{'figax' : [fig1, axis[1]],
                                                   'title' : r'Posterior Mean',
                                                   'xlabel' : 'Truth',
                                                   'ylabel' : 'Estimate'})

        # Posterior error
        true.shat_diag = np.diag(true.shat)
        self.shat_diag = np.diag(self.shat)
        fig1, axis[2], c = true.plot_comparison('shat_diag', self.shat_diag,
                                                cbar=False,
                                                **{'figax' : [fig1, axis[2]],
                                                   'title' : r'Posterior Variance',
                                                   'xlabel' : 'Truth',
                                                   'ylabel' : 'Estimate'})
        del true.shat_diag
        del self.shat_diag

        # Averaging Kernel diagonal
        true.a_diag = np.diag(true.a)
        self.a_diag = np.diag(self.a)
        fig1, axis[3], c = true.plot_comparison('a_diag', self.a_diag,
                                                cbar=True,
                                                **{'figax' : [fig1, axis[3]],
                                                   'title' : r'Averaging Kernel',
                                                   'xlabel' : 'Truth',
                                                   'ylabel' : 'Estimate'})

        for ax in fig1.axes[:-1]:
            ax.set_aspect('equal')

        # # Add colorbar
        # cax = fig1.add_axes([0.95, 0.25/2, 0.01, 0.75])
        # fig1.colorbar(c, cax=cax, boundaries=)
        # cax.tick_params(labelsize=40)
        # cax.set_ylabel('Eigenvector Value', fontsize=(BASEFONT+5)*SCALE)

        # ax[2].set_title(ax[2].get_title(), fontsize=40)

        # for a in ax.flatten():
        #     a.set_xlabel(a.get_xlabel(), fontsize=30)
        #     a.set_ylabel(a.get_ylabel(), fontsize=30)

        # Compare spectra
        fig2, ax = plt.subplots(figsize=(10*SCALE/1.25, 3*SCALE/1.25))
        fig2, ax = true.plot_info_frac(figax=[fig2, ax],
                                       label='True',
                                       color=color(0),
                                       text=False)
        self.plot_info_frac(figax=[fig2, ax],
                            label='Update',
                            ls=':',
                            color=color(5))
        ax.set_ylabel('Fraction of DOFS', fontsize=(BASEFONT+6)*SCALE,
                      labelpad=LABEL_PAD)
        ax.set_xlabel('Eigenvector Index', fontsize=(BASEFONT+6)*SCALE,
                      labelpad=LABEL_PAD)
        ax.tick_params(axis='both', which='both', labelsize=BASEFONT*SCALE)

        # Plot the first few eigenvectors to give an idea of the
        # eigenspace
        nx = 2
        ny = 4
        plot_data = [('evecs', i) for i in range(ny)]
        titles = ['%d' % (i+1) for i in range(ny)]

        kw = {'vmin' : -0.1,
              'vmax' : 0.1,
              'cmap' : 'RdBu_r',
              'add_colorbar' : False}
        cbar_kwargs = {'ticks' : [-0.1, 0, 0.1]}

        fig3, ax = plt.subplots(nx, ny, figsize=(ny*2*5.25,nx*6.75),
                               subplot_kw={'projection' : ccrs.PlateCarree()})
        plt.subplots_adjust(hspace=0.5, wspace=0.15)
        cax = fig3.add_axes([0.95, 0.25/2, 0.01, 0.75])


        for i in range(ny):
            kw['title'] = titles[i]
            kw['figax'] = [fig3, ax[0, i]]
            fig3, ax[0, i], c = true.plot_state(plot_data[i], clusters_plot,
                                                cbar=False, **kw)
            kw['figax'] = [fig3, ax[1, i]]
            fig3, ax[1, i], c = self.plot_state(plot_data[i], clusters_plot,
                                                cbar=False, **kw)

        for axis in ax.flatten():
            axis.set_title(axis.get_title(), fontsize=(BASEFONT+10)*SCALE)

        fig3.colorbar(c, cax=cax, **cbar_kwargs)
        cax.tick_params(labelsize=40)
        cax.set_ylabel('Eigenvector Value', fontsize=(BASEFONT+5)*SCALE)

        # Add label
        ax[0, 0].text(-0.1, 0.5, 'Truth', fontsize=(BASEFONT+10)*SCALE,
                      rotation=90, ha='center', va='center',
                      transform=ax[0,0].transAxes)
        ax[1, 0].text(-0.1, 0.5, 'Estimate', fontsize=(BASEFONT+10)*SCALE,
                      rotation=90, ha='center', va='center',
                      transform=ax[1,0].transAxes)

        return fig1, fig2, fig3


        # # And plot the difference between each eigenvector
        # # We use the two norm because we want to look at how close
        # # the two are and not worry so much about outliers.
        # true.plot_evec_comparison(self)
