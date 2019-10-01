import xarray as xr
import numpy as np
# from numpy import diag as diags
# from numpy import identity
from numpy.linalg import inv, norm #,eigh
from scipy.sparse import diags, identity
from scipy.linalg import eigh
from tqdm import tqdm
import copy

# clustering
from sklearn.cluster import KMeans

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams, colorbar, colors
import cartopy.crs as ccrs
import cartopy
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14
rcParams['text.usetex'] = True

class Inversion:
    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
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


    ####################################
    ### STANDARD INVERSION FUNCTIONS ###
    ####################################

    def calculate_c(self):
        self.c = self.y_base - self.k @ self.xa

    def obs_mod_diff(self, x):
        return self.y - (self.k @ x + self.c)

    def cost_func(self, x):
        cost_obs = self.obs_mod_diff(x).T \
                   @ (diags(1/(self.rf*self.so_vec))) @ self.obs_mod_diff(x)
        cost_emi = (x - self.xa).T @ diags(1/self.sa_vec) @ (x - self.xa)
        print('     Cost function: %.2f' % (cost_obs + cost_emi))
        return cost_obs + cost_emi

    def solve_inversion(self):
        so_inv = diags(1/(self.rf*self.so_vec))
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
        print('     DOFS: %.2f' % np.trace(self.a))

        # Calculate the new set of modeled observations.
        print('Calculating updated modeled observations.')
        self.y_out = self.k @ self.xhat + self.c

        print('Inversion complete.')

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

    def plot_state(self, attribute, clusters_plot, default_value=0, **kw):
        # Get the data from the attribute argument. The attribute argument
        # is either a string or a string and int. ## EXPAND COMMENT HERE
        try:
            attribute, selection = attribute
            data = getattr(self, attribute)[:, selection]
            attribute_str = attribute + '_' + str(selection)
        except ValueError:
            data = getattr(self, attribute)
            attribute_str = attribute

        # Force the data to have dimension equal to the state dimension
        assert data.shape[0] == self.nstate, \
               'Dimension mismatch: Data does not match state dimension.'

        # Match the data to lat/lon data
        data = self.match_data_to_clusters(data, clusters_plot, default_value)
        
        # Get kw
        try: 
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(20,8),
                                   subplot_kw={'projection' : 
                                               ccrs.PlateCarree()})

        title = kw.pop('title', attribute_str)
        kw['cmap'] = kw.get('cmap', 'viridis')
        kw['vmin'] = kw.get('vmin', data.min())
        kw['vmax'] = kw.get('vmax', data.max())
            
        c = data.plot(ax=ax, snap=True, **kw)

        ax.set_title(title, y=1.07, fontsize=24)
        ax.add_feature(cartopy.feature.OCEAN, facecolor='0.98')
        ax.add_feature(cartopy.feature.LAND, facecolor='0.98')
        ax.coastlines(color='grey')
        ax.gridlines(linestyle=':', draw_labels=True, color='grey')

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

        fig, ax = plt.subplots(nx, ny, figsize=(ny*2*5.25,nx*6.75),
                               subplot_kw={'projection' : ccrs.PlateCarree()})
        plt.subplots_adjust(hspace=0.3, wspace=0.15)
        cax = fig.add_axes([0.95, 0.25/2, 0.01, 0.75])
        
        kw['add_colorbar'] = False
        cbar_kwargs = kw.pop('cbar_kwargs', {})

        for i, axis in enumerate(ax.flatten()):
            kw['figax'] = [fig, axis]
            try:
                kw['title'] = titles[i]
                kw['vmin'] = vmins[i]
                kw['vmax'] = vmaxs[i]
            except NameError:
                pass
            fig, axis, c = self.plot_state(attributes[i], clusters_plot, **kw)

        fig.colorbar(c, cax=cax, **cbar_kwargs)
        cax.tick_params(labelsize=16)

        return fig, ax


class ReducedRankInversion(Inversion):
    # class variables shared by all instances

    def __init__(self, k, xa, sa_vec, y, y_base, so_vec):
        # We inherit from the inversion class and create space for
        # the reduced rank solutions.
        Inversion.__init__(self, k, xa, sa_vec, y, y_base, so_vec)

        # We create space for the rank
        self.rank = None

        # We also want to save the eigendecomposition values
        self.evals = None
        self.evecs = None

    ########################################
    ### REDUCED RANK INVERSION FUNCTIONS ###
    ########################################

    def pph(self):
        # Calculate the prior pre-conditioned Hessian
        sa_sqrt = diags(self.sa_vec**0.5)
        pph_m  = (sa_sqrt @ self.k.T) \
                 @ diags(1/self.so_vec) @ (sa_sqrt @ self.k.T).T
        print('Calculated PPH.')
        return pph_m

    def edecomp(self):
        # Perform the eigendecomposition of the prior 
        # pre-conditioned Hessian
        # We return the evals of the projection, not of the 
        # prior pre-conditioned Hessian.
        evals, evecs = eigh(self.pph())
        print('Eigendecomposition complete.')
        
        # Sort evals and evecs by eval
        evals = evals[::-1]
        evecs = evecs[:,::-1]

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
        self.evals = evals/(1 + evals)
        self.evecs = evecs

    def projection(self, pct_of_info=None, rank=None):
        # Conduct the eigendecomposition of the prior pre-conditioned
        # Hessian
        if (self.evals is None) or (self.evecs is None):
            print('Performing eigendecomposition.')
            self.edecomp()

        # Subset the evecs according to the rank provided.
        if sum(x is not None for x in [pct_of_info, rank]) > 1:
            print('pct_of_info = ', pct_of_info)
            print('rank = ', rank)
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank]) == 0:
            raise AttributeError('Must provide one of pct_of_info or rank.')
        elif pct_of_info is not None:
            rank = np.argwhere(self.evals < (1 - pct_of_info))[0][0]
            print('Calculated rank from rank threshold: %d' % rank)
        elif rank is not None:
            print('Using defined rank: %d' % rank)

        evecs_subset = self.evecs[:,:rank]

        # Calculate the prolongation and reduction operators and
        # the resulting projection operator.
        prolongation = (evecs_subset.T * self.sa_vec**0.5).T
        reduction = (1/self.sa_vec**0.5) * evecs_subset.T
        projection = prolongation @ reduction

        return rank, prolongation, reduction, projection

##### finish converting reduced rank code over ######

    ##########################
    ########## ERROR #########
    ##########################

    def calc_error(self, true, attribute):
        e = getattr(self, attribute)
        t = getattr(true, attribute)
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
            fig, ax = plt.subplots(figsize=(10, 5))
        label = kw.pop('label', '')
        color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
        ls = kw.pop('ls', '-')
        if kw:
            raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

        ax.plot(self.evals, label=label, c=color, ls=ls)

        ax.set_facecolor('0.98')
        ax.legend(frameon=False, fontsize=22)
        ax.set_xlabel('Eigenvalue Index', fontsize=22)
        ax.set_ylabel(r'Q$_{DOF}$ Eigenvalues', fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)

        return fig, ax

    def plot_evec_comparison(self, compare_data, **kw):
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(10, 5))
        label = kw.pop('label', '')
        color = kw.pop('color', plt.cm.get_cmap('inferno')(5))
        ls = kw.pop('ls', '-')
        if kw:
            raise TypeError('Unexpected kwargs provided: %s' % list(kw.keys()))

        diff_neg = np.linalg.norm(self.evecs - compare_data.evecs, axis=0)
        diff_pos = np.linalg.norm(self.evecs + compare_data.evecs, axis=0)
        rel_diff = np.minimum(diff_neg, diff_pos)/np.linalg.norm(self.evecs, axis=0)
        ax.plot(rel_diff, label=label, c=color, ls=ls)

        ax.set_facecolor('0.98')
        ax.set_xlabel('Eigenvalue Index', fontsize=22)
        ax.set_ylabel(r'Eigenvector Difference', fontsize=22)
        ax.set_ylim(0,3)
        plt.tick_params(axis='both', which='major', labelsize=18)

        return fig, ax

    def plot_comparison(self, attribute, compare_data, **kw):
        # get attribute data
        xdata = getattr(self, attribute)

        # Get plotting kwargs
        try:
            fig, ax = kw.pop('figax')
        except KeyError:
            fig, ax = plt.subplots(figsize=(10, 10))

        xlabel = kw.pop('xlabel', 'True ' + attribute)
        ylabel = kw.pop('ylabel', 'Estimated ' + attribute)
        title = kw.pop('title', 'Estimated vs. True ' + attribute)

        if type(compare_data) == dict:
           # We need to know how many data sets were passed
            n = len(compare_data)
            cmap = kw.pop('cmap', plt.cm.get_cmap('inferno', lut=n))
            cbar_ticklabels = kw.pop('cbar_ticklabels', 
                                     list(compare_data.keys()))
            cbar_title = kw.pop('cbar_title', '')

            # Plot data
            count = 0
            for k, ydata in compare_data.items():
                ax.scatter(xdata, ydata, 
                           alpha=0.5, s=5, c=cmap(count))
                count += 1

            # Color bar
            cax = fig.add_axes([0.95, 0.25/2, 0.01, 0.75])
            norm = colors.Normalize(vmin=0, vmax=n)
            cb = colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
            cb.set_ticks(0.5 + np.arange(0,n+1))
            cb.set_ticklabels(cbar_ticklabels)
            cb.set_label(cbar_title, fontsize=16)
            plt.tick_params(axis='both', which='both', labelsize=16)

        else:
            c = kw.pop('color', 
                       np.asarray(plt.cm.get_cmap('inferno', 
                                                  lut=10)(3)).reshape(1,-1))
            ax.scatter(xdata, compare_data, 
                       alpha=0.5, s=5, c=c)

        # Aesthetics
        ax.set_facecolor('0.98')
        
        try:
            ylim = kw.pop('ylim')
            xlim = kw.pop('xlim')
            xy = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
        except:
            lims = ax.axis()
            xlim = ylim = xy = (min(lims[0], lims[2]), max(lims[1], lims[3]))

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.plot(xy, xy, c='0.1', lw=2, ls=':', alpha=0.5, zorder=0)
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.tick_params(axis='both', which='both', labelsize=16)

        return fig, ax

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
        sig_idx = significance.argsort()[::-1]

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

    @staticmethod
    def calculate_perturbation_matrix(state_vector, significance):
        sv_sig = pd.DataFrame({'sv' : state_vector, 
                               'sig' : significance})
        sv_sig = sv_sig.groupby(by='sv').mean().reset_index()
        sv_sig = sv_sig.sort_values(by='sig')
        sv_sig = sv_sig[sv_sig['sv'] > 0]
        p = np.zeros((len(state_vector), len(sv_elements)))
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

#################################################################
##### Rewrite this function to accept either pct_of_info or rank
#################################################################
    def calculate_significance(self, 
                               pct_of_info=None, rank=None, prolongation=None):
        if prolongation is None:
            rank, prolongation, _, _ = self.projection(pct_of_info=pct_of_info, 
                                                       rank=rank)
        significance = np.sqrt((prolongation**2)).sum(axis=1)
        return rank, significance

    def broyden(self, forward_model, perturbation_matrix):
        for p in perturbation_matrix.T:
            p = p.reshape(-1, 1)
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

        # Retrieve the prolongation operator associated with
        # this instance of the Jacobian for the rank specified
        # by the function call. These are the eigenvectors
        # that we will perturb.
        # Calculate the weight of each full element space as sum
        # of squares
        new.rank, significance = self.calculate_significance(pct_of_info)

        # If previously optimized, set significance to 0
        if len(self.perturbed_cells) > 0:
            print('Ignoring previously optimized grid cells.')
            significance[self.perturbed_cells] = 0

        # We need the new state vector first. This gives us the
        # clusterings of the base resolution state vector
        # elements as dictated by n_cells and n_cluster_size.
        state_vector = self.aggregate_cells_kmeans(clusters_plot, significance, 
                                                   n_cells, n_cluster_size)

        # And now get the perturbation matrix
        perturbation_matrix = self.calculate_perturbation_matrix(state_vector)
        
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

    # def update_jacobian_ag(self, forward_model, clusters_plot,
    #                     pct_of_info=None,
    #                     n_cells=[100, 200],
    #                     n_cluster_size=[1, 2],
    #                     k_base=None):#, #threshold=0,
    #                     # perturbation_factor=0.5):

    #     # We start by creating a new instance of the class. This
    #     # is where 
    #     new = ReducedRankJacobian(k=copy.deepcopy(self.k),
    #                               xa=copy.deepcopy(self.xa),
    #                               sa_vec=copy.deepcopy(self.sa_vec),
    #                               y=copy.deepcopy(self.y),
    #                               y_base=copy.deepcopy(self.y_base),
    #                               so_vec=copy.deepcopy(self.so_vec))
    #     new.model_runs = copy.deepcopy(self.model_runs)
    #     new.rf = copy.deepcopy(self.rf)

    #     # Retrieve the prolongation operator associated with
    #     # this instance of the Jacobian for the rank specified
    #     # by the function call. These are the eigenvectors
    #     # that we will perturb.
    #     # Calculate the weight of each full element space as sum
    #     # of squares
    #     new.rank, significance = self.calculate_significance(pct_of_info)

    #     # If previously optimized, set significance to 0
    #     if len(self.perturbed_cells) > 0:
    #         print('Ignoring previously optimized grid cells.')
    #         significance[self.perturbed_cells] = 0

    #     # We need the new state vector first. This gives us the
    #     # clusterings of the base resolution state vector
    #     # elements as dictated by n_cells and n_cluster_size.
    #     state_vector = self.aggregate_cells_kmeans(clusters_plot, significance, 
    #                                                n_cells, n_cluster_size)
        
    #     # This gives us the number of model runs.
    #     new.model_runs += len(np.unique(state_vector)[1:])

    #     # Find the individual grid cells that are perturbed
    #     counts = np.unique(state_vector, return_counts=True)
    #     new_perturbed_cells = counts[0][counts[1] == 1]
    #     new.perturbed_cells = np.append(self.perturbed_cells,
    #                                     new_perturbed_cells).astype(int)

    #     # Create space for the new Jacobian
    #     if k_base is None:
    #         k_new = copy.deepcopy(self.k)
    #         # k_new = np.zeros(self.k.shape)
    #     else:
    #         k_new = k_base

    #     # Now update the Jacobian
    #     k_new = self.calculate_k(forward_model, state_vector, k_new)
    #     new.k = k_new

    #     # And do the eigendecomposition
    #     new.edecomp()

    #     # And solve the inversion
    #     new.solve_inversion()

    #     return new

    # Rewrite to take any prolongation matrix?
    def update_jacobian_br(self, forward_model,
                           pct_of_info=None, rank=None, prolongation=None,
                           k_base=None):
        # We start by creating a new instance of the class.
        if k_base is None:
            k_base = copy.deepcopy(self.k)
        new = ReducedRankJacobian(k=k_base,
                                  xa=copy.deepcopy(self.xa),
                                  sa_vec=copy.deepcopy(self.sa_vec),
                                  y=copy.deepcopy(self.y),
                                  y_base=copy.deepcopy(self.y_base),
                                  so_vec=copy.deepcopy(self.so_vec))
        new.model_runs = copy.deepcopy(self.model_runs)
        new.rf = copy.deepcopy(self.rf)

        # Retrieve the prolongation operator associated with
        # this instance of the Jacobian for the rank specified
        # by the function call. These are the eigenvectors
        # that we will perturb.
        # if (pct_of_info is not None) and (rank is not None):
        if sum(x is not None for x in [pct_of_info, rank, prolongation]) > 1:
            raise AttributeError('Conflicting arguments provided to determine rank.')
        elif sum(x is not None for x in [pct_of_info, rank, prolongation]) == 0:
            raise AttributeError('Must provide one of pct_of_info, rank, or prolongation.')
        elif prolongation is not None:
            print('Using given prolongation.')
        else:
            _, prolongation, _, _ = self.projection(pct_of_info=pct_of_info,
                                                    rank=rank)

        # Update the number of model runs
        new.model_runs += prolongation.shape[1]

        # We update the Jacobian perturbation-by-perturbation following
        # Broyden's theorem.
        new.broyden(forward_model, 
                    perturbation_matrix=prolongation)
        # for column in prolongation.T:
        #     col = column.reshape(-1, 1)
        #     new.k += ((forward_model - new.k) @ col @ col.T)/ \
        #              np.linalg.norm(col)**2

        # Update the value of c in the new instance
        new.calculate_c()

        # And do the eigendecomposition
        new.edecomp()

        # And solve the inversion
        new.solve_inversion()

        return new


    def full_analysis(self, true, clusters_plot):
        # n_model_runs = (np.array(n_cells)/np.array(n_cluster_size)).sum()
        # update = self.update_jacobian(forward_model=true.k,
        #                               clusters_plot=clusters_plot, 
        #                               pct_of_info=pct_of_info,
        #                               n_cells=n_cells,
        #                               n_cluster_size=n_cluster_size)
        # update.edecomp()
        # # IS THE RANK HERE CORRECT?
        # rank, prolong, reduct, project = true.projection(rank=n_model_runs)
        # true.k_reduced = true.k @ prolong
        # rank, prolong, reduct, project = update.projection(rank=n_model_runs)
        # update.k_reduced = update.k @ prolong

        # Compare full and reduced dimension jacobians
        fig1, ax = plt.subplots(1, 3, figsize=(33, 10))
        k_label = r'$\frac{\vert\vert K_{true} - K_{update} \vert\vert}{\vert\vert K_{true} \vert\vert}$'
        x_label = r'$\frac{\vert\vert \hat{x}_{true} - \hat{x}_{update} \vert\vert}{\vert\vert \hat{x}_{true} \vert\vert}$'
        s_label = r'$\frac{\vert\vert \hat{S}_{true} - \hat{S}_{update} \vert\vert}{\vert\vert \hat{S}_{true} \vert\vert}$'
        
        # Full dimension Jacobian
        true.plot_comparison('k', self.k, **{'figax' : [fig1, ax[0]]})
        err_abs, err_rel = self.calc_error(true, 'k')
        ax[0].text(0.05, 0.9, 
                   r'%s = %.2f' % (k_label, err_rel), 
                   fontsize=25,
                   transform=ax[0].transAxes)

        # Posterior
        true.plot_comparison('xhat', self.xhat, **{'figax' : [fig1, ax[1]]})
        err_abs, err_rel = self.calc_error(true, 'xhat')
        ax[1].text(0.05, 0.9, 
                   r'%s = %.2f' % (x_label, err_rel), 
                   fontsize=25,
                   transform=ax[1].transAxes)

        # Posterior error
        true.plot_comparison('shat', self.shat, **{'figax' : [fig1, ax[2]]})
        err_abs, err_rel = self.calc_error(true, 'shat')
        ax[2].text(0.05, 0.9, 
                   r'%s = %.2f' % (s_label, err_rel), 
                   fontsize=25,
                   transform=ax[2].transAxes)

        # # Reduced dimension Jacobian
        # true.plot_comparison('k_reduced', self.k_reduced, 
        #                      **{'figax' : [fig, ax[1]]})

        # Compare spectra
        c = plt.cm.get_cmap('inferno', lut=10)
        fig2, ax = plt.subplots(figsize=(20, 5))
        fig2, ax = true.plot_eval_spectra(figax=[fig2, ax],
                                          label='True',
                                          color=c(0))
        self.plot_eval_spectra(figax=[fig2, ax],
                               label='Update',
                               ls=':',
                               color=c(5))

        # Plot the first few eigenvectors to give an idea of the
        # eigenspace
        nx = 2
        ny = 5
        plot_data = [('evecs', i) for i in range(nx*ny)]
        titles = ['Eigenvector %d' % (i+1) for i in range(nx*ny)]
        kw = {'titles' : titles,
              'vmin' : -0.1,
              'vmax' : 0.1,
              'cmap' : 'RdBu_r',
              'cbar_kwargs' : {'ticks' : [-0.1, 0, 0.1]}}

        fig3, ax = true.plot_grid(plot_data, 
                                 nx=nx, ny=ny, 
                                 clusters_plot=clusters_plot,
                                 **kw)
        plt.suptitle('Eigenvectors (True Jacobian)',
                     fontsize=40, y=1.05)

        fig4, ax = self.plot_grid(plot_data, 
                                  nx=nx, ny=ny,
                                  clusters_plot=clusters_plot, 
                                  **kw)
        plt.suptitle('Eigenvectors (Updated Jacobian, %d model runs)' 
                     % self.model_runs,
                     fontsize=40, y=1.05);

        # And plot the difference between each eigenvector
        # We use the two norm because we want to look at how close
        # the two are and not worry so much about outliers.
        true.plot_evec_comparison(self)


    # def update_jacobian(self, forward_model, clusters_plot,
    #                     pct_of_info=None,
    #                     k_base=None):#, #threshold=0,
    #                     # perturbation_factor=0.5):

    #     # We start by creating a new instance of the class. This
    #     # is where 
    #     new = ReducedRankJacobian(k=self.k,
    #                               xa=self.xa,
    #                               sa_vec=self.sa_vec,
    #                               y=self.y,
    #                               y_base=self.y_base,
    #                               so_vec=self.so_vec)

    #     # Retrieve the prolongation operator associated with
    #     # this instance of the Jacobian for the rank specified
    #     # by the function call. These are the eigenvectors
    #     # that we will perturb.
    #     # Calculate the weight of each full element space as sum
    #     # of squares
    #     new.rank, significance = self.calculate_significance(pct_of_info)

    #     # If previously optimized, set significance to 0
    #     if len(self.perturbed_cells) > 0:
    #         print('Ignoring previously optimized grid cells.')
    #         significance[self.perturbed_cells] = 0

    #     # Find the n=rank most significant grid cells. These are the
    #     # grid cells we will perturb.
    #     perturbation_idx = significance.argsort()[::-1][:new.rank]
    #     new.perturbed_cells = np.append(self.perturbed_cells, 
    #                                     perturbation_idx).astype(int)
    #     new.model_runs = len(new.perturbed_cells)

    #     # Create space for the new Jacobian
    #     if k_base is None:
    #         k_new = self.k
    #         # k_new = np.zeros(self.k.shape)
    #     else:
    #         k_new = k_base

    #     # And we will simply retrieve the forward model for those
    #     # columns (this is equivalent). However, we want to take 
    #     # advantage of proximity effects--if the perturbed state 
    #     # element is close to other, unperturbed elements, we would
    #     # rather use some fraction of the model response from the 
    #     # nearby perturbation to estimate its model response
    #     # (wow rewrite this terrible)
    #     for sv in perturbation_idx:
    #         # Get the Jacobian for that state vector element
    #         perturbation = forward_model[:, sv].reshape(-1, 1)

    #         # Get the indices of the neighboring grid cells
    #         idx = self.get_neighboring_cells(sv, clusters_plot)

    #         # Check if we are otherwise optimizing any of these grid
    #         # cells. Specifically, find the cells that are in
    #         # neighbor_idx but not in perturbation_idx nor that were
    #         # previously optimized.
    #         idx = np.setdiff1d(idx, new.perturbed_cells, 
    #                            assume_unique=True)

    #         # Add sv back in.
    #         idx = np.append(idx, sv)

    #         # And then update k_new
    #         k_new[:,idx] = np.tile(perturbation, [1, len(idx)])

    #     new.k = k_new
    #     return new

        # # Apply the forward model (in this case the true Jacobian)
        # # to the perturbations, given by the prolongation matrix.
        # k_reduced = forward_model @ prolongation

        # # We now solve for the updated Jacobian by minimizing
        # # the derivative of the reduced-rank Jacobian
        # # at every entry
        # k_new = np.zeros(self.k.shape)
        # # pbar = tqdm(total=len(k_new.flatten())/5000)
        # # count = 0
        # for i in range(k_new.shape[0]):
        #     for j in range(k_new.shape[1]):
        #         #     pbar.update(1)
        #         numerator = (k_reduced[i,:]*prolongation[j,:]).sum()
        #         denominator = (prolongation[j,:]**2).sum()

        #         # If the denominator is below some threshold, it signals
        #         # that there is insufficient information content. We 
        #         # opt only to optimize the Jacobian value in cases where 
        #         # information content is above that threshold.
        #         # (The denominator is the sum of squares of the row of the 
        #         # prolongation operator in question. It is a measure
        #         # of the total influence of the reduced space on an
        #         # individual element of the full state space. 
        #         # If small, then that element is not spanned by the reduced
        #         # space elements.)
        #         # By default, the threshold is 0.
        #         if np.sqrt(denominator) > threshold:
        #             k_new[i,j] = numerator/denominator

        # return k_new

