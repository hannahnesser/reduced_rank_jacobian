import xarray as xr
import numpy as np
import math

# import sys
# sys.path.append('.')
# import jacobian


def open_clusters(cluster_path):
    c = xr.open_dataset(cluster_path)['Clusters']
    c = c.drop('time')
    c = c.squeeze('time')
    return c


def reduced_clusters(full_clusters):
    # reduce from the full 0.5 x 0.625 grid to that which we are working on
    # this operation should be superficial (i.e. for plotting inversion results)

    # Find the reduction factor
    n = find_grouping_factor(full_clusters)

    # Create a reduced grid
    rc = np.zeros([math.floor(d/n) for d in full_clusters.shape])
    rc_lat = np.zeros(rc.shape[0])
    rc_lon = np.zeros(rc.shape[1])
    for i in range(len(rc_lon)):
        for j in range(len(rc_lat)):
            rc_lat[j] = full_clusters.coords['lat'][(n*j):(n*(j+1))].mean()
            rc_lon[i] = full_clusters.coords['lon'][(n*i):(n*(i+1))].mean()
            rc[j,i] = full_clusters[(n*j),(n*i)]

    # convert to xarray
    return xr.DataArray(rc, coords=[rc_lat, rc_lon], dims=['lat', 'lon'])

def reduce_SV_elements(clusters_res1, clusters_res2, delta_obs=None):
    # We wish to only use clusters that are present in full at both resolutions
    c1 = clusters_res1.copy()
    c2 = clusters_res2.copy()

    # We want c2 to be the coarser resolution
    n1 = find_grouping_factor(c1)
    n2 = find_grouping_factor(c2)
    if n2 < n1: # If c1 is coarser than c2
        temp = c1
        c1 = c2
        c2 = temp

    # Recall delta_obs is at the finest resolution only.
    if delta_obs is not None:
        # Find the total effect of every perturbation experiment. Where these are equal
        # to 0, the perturbation had no effect on the observations, and
        # we want to eliminate those state vector elements.

        # We can only use the original cluster mapping, since that is what
        # matches the delta_obs matrix. So, if the number of clusters in the
        # finest resolution cluster xarray (c2) does not match the NSV dimension
        # of delta_obs, we exit the routine.
        if c1.max() != len(delta_obs.coords['NSV']):
            sys.exit()
        else:
            zero_coords_NSV = delta_obs.coords['NSV'].where(delta_obs.sum(dim='Nobs') == 0, drop=True).values

            # Set the finest resolution clusters equal to 0 where the perturbation
            # had no effect on observations
            c1.values[c1.isin(zero_coords_NSV)] = 0

            # And then eliminate those SV elements from delta_obs, too
            delta_obs = delta_obs.where(delta_obs.sum(dim='Nobs') > 0, drop=True)

    # Now match up the clusters: if either of the resolutions has a zero value
    # at a point, then both final cluster files should be zero
    c1_s = c1.copy()
    c1_s.values[(c1 == 0) | (c2 == 0)] = 0

    c2_s = c2.copy()
    c2_s.values[(c1 == 0) | (c2 == 0)] = 0

    # drop incomplete squares (i.e. if there are not n boxes with values > 0
    # in one of the larger grid boxes (c2), then set all the boxes at both
    # resolutions to 0)
    for i, c in enumerate(np.unique(c2)[1:]):
        if c1.where(c2 == c, drop=True).min() == 0:
            c2_s.values[c2 == c] = 0
            c1_s.values[c2 == c] = 0

    # Finally, we will alter delta_obs one last time (if relevant)
    if delta_obs is not None:
        c1_s_SV_elements = np.unique(c1_s)[1:]
        delta_obs = delta_obs.where(delta_obs.coords['NSV'].isin(c1_s_SV_elements), drop=True)

        # Because we have now reduced the state vector dimension, there may be observations that
        # are unaffected by the remaining state vector elements. We eliminate those one last time.
        delta_obs = delta_obs.where(delta_obs.sum(dim='NSV') > 0, drop=True)

        # then return everything
        return c1_s, c2_s, delta_obs
    else:
        return c1_s, c2_s

def find_grouping_factor(clusters):
    _, n = [t[1:] for t in np.unique(clusters, return_counts=True)]
    n = np.unique(n)
    if len(n) == 1:
        n = np.sqrt(n)[0]
    else:
        sys.exit()
    return int(n)

def find_cluster_mapping(clusters1, clusters2):
    # We want c2 to be the coarser resolution
    # c1: finer resolution
    # c2: coarser resolution
    n1 = find_grouping_factor(clusters1)
    n2 = find_grouping_factor(clusters2)
    if n2 < n1: # If c1 is coarser than c2
        temp = clusters1
        clusters1 = clusters2
        clusters2 = temp

    # Join in the clusters files
    cs = clusters1.to_dataset(name='1')
    cs['2'] = clusters2
    cs = cs.to_dataframe().reset_index(drop=True)#.drop(['lat', 'lon'], axis=1)
    cs = cs[cs['1'] > 0]
    cs = cs.sort_values(by=['2', '1'])
    cs = cs.drop_duplicates()

    return cs

if __name__ == '__main__':
    from os.path import join
    from os import listdir
    import sys

    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = 'AppleGothic'
    rcParams['font.size'] = 14

    input_dir = str(sys.argv[1])
    c_in = str(sys.argv[2])
    c_str = str(sys.argv[3])

    names = [c_str, c_str + '_plot']
    names = ['clusters_' + n + '.nc' for n in names]
    if not set(names).issubset(listdir(input_dir)):
        print('Clusters do not yet exist.')
        # Import clusters
        c = open_clusters(join(input_dir, c_in + '_' + c_str + '.nc'))

        # Create reduced clusters (i.e. our actual grid) for plotting purposes
        c_r = reduced_clusters(c)

        # Save out final clusters.
        c.to_netcdf(join(input_dir, names[0]))
        c_r.to_netcdf(join(input_dir, names[1]))
    else:
        print('Clusters already exist.')



# Routine if we need larger clusters to regrid from a coarse grid Jacobian--but
# this is not how we are estimating the Jacobian this time.
# if __name__ == '__main__':
#     from os.path import join
#     from os import listdir
#     import sys

#     import matplotlib.pyplot as plt
#     from matplotlib import rcParams
#     rcParams['font.family'] = 'sans-serif'
#     rcParams['font.sans-serif'] = 'Arial'
#     rcParams['font.size'] = 14

#     input_dir = str(sys.argv[1])
#     c_in = str(sys.argv[2])
#     c_small_str = str(sys.argv[3])
#     c_large_str = str(sys.argv[4])

#     names = [c_small_str, c_large_str, c_small_str + '_plot']
#     names = ['clusters_' + n + '.nc' for n in names]
#     if not set(names).issubset(listdir(input_dir)):
#         print('Clusters do not yet exist.')
#         # Import clusters
#         c_small = open_clusters(join(input_dir, c_in + '_' + c_small_str + '.nc'))
#         c_large = open_clusters(join(input_dir, c_in + '_' + c_large_str + '.nc'))

#         # We only want to work in the domain where we have full clusters
#         # at both resolutions
#         c_small_s, c_large_s = reduce_SV_elements(c_small, c_large)

#         # Create reduced clusters (i.e. our actual grid) for plotting purposes
#         c_small_r = reduced_clusters(c_small_s)

#         # Confirm that the new clusters (_s) match at both resolutions
#         fig, ax = plt.subplots(1,3,figsize=(15,3))
#         plt.subplots_adjust(wspace=0.4)
#         c_small.where(c_small > 0).plot(ax=ax[0])
#         c_small_s.where(c_small_s > 0).plot(ax=ax[1])
#         c_large_s.where(c_large_s > 0).plot(ax=ax[2])
#         # plt.show()
#         plt.savefig(join(input_dir, 'clusters.png'))

#         # And confirm numerically.
#         nc_small = len(np.unique(c_small_s))-1
#         nc_large = len(np.unique(c_large_s))-1
#         print('    Number of %s Clusters: %d' % (c_small_str, nc_small))
#         print('    Number of %s Clusters: %d' % (c_large_str, nc_large))
#         print('    Ratio of %s Clusters to %s Clusters: %.1f' % (c_large_str, c_small_str, (nc_small/nc_large)))

#         # Save out final clusters.
#         c_small_s.to_netcdf(join(input_dir, names[0]))
#         c_large_s.to_netcdf(join(input_dir, names[1]))
#         c_small_r.to_netcdf(join(input_dir, names[2]))
#     else:
#         print('Clusters already exist.')
