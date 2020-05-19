def f(x, y):
    x, y = np.meshgrid(x, y)
    return (1 - x / 2 + x**5 + y**3 + x*y**2) * np.exp(-x**2 -y**2)


import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import itertools
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.collections import PolyCollection, LineCollection
import numpy as np


def add_contourf3d(ax, contour_set):
    proj_ax = contour_set.collections[0].axes
    for zlev, collection in zip(contour_set.levels, contour_set.collections):
        paths = collection.get_paths()
        # Figure out the matplotlib transform to take us from the X, Y
        # coordinates to the projection coordinates.
        trans_to_proj = collection.get_transform() - proj_ax.transData

        paths = [trans_to_proj.transform_path(path) for path in paths]
        verts = [path.vertices for path in paths]
        codes = [path.codes for path in paths]
        pc = PolyCollection([])
        pc.set_verts_and_codes(verts, codes)

        # Copy all of the parameters from the contour (like colors) manually.
        # Ideally we would use update_from, but that also copies things like
        # the transform, and messes up the 3d plot.
        pc.set_facecolor(collection.get_facecolor())
        pc.set_edgecolor(collection.get_edgecolor())
        pc.set_alpha(collection.get_alpha())

        ax3d.add_collection3d(pc, zs=zlev)

    # Update the limit of the 3d axes based on the limit of the axes that
    # produced the contour.
    proj_ax.autoscale_view()

    ax3d.set_xlim(*proj_ax.get_xlim())
    ax3d.set_ylim(*proj_ax.get_ylim())
    ax3d.set_zlim(Z.min(), Z.max())

def add_feature3d(ax, feature, clip_geom=None, zs=None):
    """
    Add the given feature to the given axes.
    """
    concat = lambda iterable: list(itertools.chain.from_iterable(iterable))

    target_projection = ax.projection
    geoms = list(feature.geometries())

    if target_projection != feature.crs:
        # Transform the geometries from the feature's CRS into the
        # desired projection.
        geoms = [target_projection.project_geometry(geom, feature.crs)
                 for geom in geoms]

    if clip_geom:
        # Clip the geometries based on the extent of the map (because mpl3d
        # can't do it for us)
        geoms = [geom.intersection(clip_geom) for geom in geoms]

    # Convert the geometries to paths so we can use them in matplotlib.
    paths = concat(geos_to_path(geom) for geom in geoms)

    # Bug: mpl3d can't handle edgecolor='face'
    kwargs = feature.kwargs
    if kwargs.get('edgecolor') == 'face':
        kwargs['edgecolor'] = kwargs['facecolor']

    polys = concat(path.to_polygons(closed_only=False) for path in paths)

    if kwargs.get('facecolor', 'none') == 'none':
        lc = LineCollection(polys, **kwargs)
    else:
        lc = PolyCollection(polys, closed=False, **kwargs)
    ax3d.add_collection3d(lc, zs=zs)

nx, ny = 256, 512
X = np.linspace(-180, 10, nx)
Y = np.linspace(-89, 89, ny)
Z = f(np.linspace(-3, 3, nx), np.linspace(-3, 3, ny))


fig = plt.figure()
ax3d = fig.add_axes([0, 0, 1, 1], projection='3d')

# Make an axes that we can use for mapping the data in 2d.
proj_ax = plt.figure().add_axes([0, 0, 1, 1], projection=ccrs.Robinson())
cs = proj_ax.contourf(X, Y, Z, transform=ccrs.PlateCarree(), alpha=1)

ax3d.projection = proj_ax.projection
add_contourf3d(ax3d, cs)

# Use the convenience (private) method to get the extent as a shapely geometry.
clip_geom = proj_ax._get_extent_geom().buffer(0)
print(clip_geom)

zbase = ax3d.get_zlim()[0]
add_feature3d(ax3d, cartopy.feature.OCEAN, clip_geom, zs=zbase)
add_feature3d(ax3d, cartopy.feature.LAND, clip_geom, zs=zbase)
add_feature3d(ax3d, cartopy.feature.COASTLINE, clip_geom, zs=zbase)

add_feature3d(ax3d, cartopy.feature.OCEAN, clip_geom, zs=0.5)
add_feature3d(ax3d, cartopy.feature.LAND, clip_geom, zs=0.5)
add_feature3d(ax3d, cartopy.feature.COASTLINE, clip_geom, zs=0.5)

# Put the outline (neatline) of the projection on.
outline = cartopy.feature.ShapelyFeature(
    [proj_ax.projection.boundary], proj_ax.projection,
    edgecolor='black', facecolor='none')
add_feature3d(ax3d, outline, clip_geom, zs=zbase)


# Close the intermediate (2d) figure
plt.close(proj_ax.figure)
plt.show()
