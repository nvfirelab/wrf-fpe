import argparse
import glob

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely import plotting as shplt
from tqdm.auto import tqdm
import cartopy.crs as ccrs

DEFAULT_BACKGROUND_COLOR = 'white'
DEFAULT_FOREGROUND_COLOR = 'black'

def get_shape_bounds(shape_path) -> list[float, float, float, float]:

    shp = gpd.read_file(shape_path)
    return shp.total_bounds

def define_fig(bounds):
    
    fig = plt.figure(figsize=(22,16))
    ax = plt.axes(projection = ccrs.PlateCarree())

    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], 
                  crs=ccrs.PlateCarree())

    return fig, ax

# def normalize_polygon(shape: shapely.geometry.polygon.Polygon, 
# 					  max_bound=1):
#     crd = shape.exterior.coords
#     crd_x = np.array([pt[0] for pt in crd])
#     crd_y = np.array([pt[1] for pt in crd])

#     min_x = np.min(crd_x)
#     max_x = np.max(crd_x)
#     min_y = np.min(crd_y)
#     max_y = np.max(crd_y)

#     shape_n = shapely.transform(shape, 
#                                 lambda x, y: (((x-min_x)/(max_x-min_x)), ((y-min_y)/(max_y-min_y))), 
#                                 interleaved=False)
    
#     shape_n_scaled = shapely.transform(shape_n, 
#                                        lambda x: x * max_bound)

#     return shape_n_scaled

def create_image_from_poly(shape, 
                           save_directory,
                           image_name, 
                           bounds,
                           background_color: str = DEFAULT_BACKGROUND_COLOR, 
                           foreground_color: str = DEFAULT_FOREGROUND_COLOR, 
                           fill_shape: bool = True) -> str:

    fig, ax = define_fig(bounds)

    union_shapes = shape.union_all()

    shplt.plot_polygon(union_shapes, 
                       add_points=False,
                       facecolor=foreground_color if fill_shape else background_color,
                       edgecolor=foreground_color,
                       ax=ax)

    ax.set_facecolor(background_color)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    output_image_path = save_directory + '/' + image_name + '.jpg'
    fig.savefig(output_image_path, 
                bbox_inches='tight', 
                pad_inches=0)
    plt.close(fig) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=f'Convert a polygon from a shape file to an image')
    parser.add_argument('input_file_directory',  
                        help='directory to read input files from',
                        type=str)
    parser.add_argument('save_directory',  
                        help='directory to save output images to',
                        type=str)
    parser.add_argument('--last-shapefile',
                        help='specify the last shapefile (i.e. with largest extent) to get the total bounds from',
                        type=str,
                        metavar=('color'),
                        dest='largest_perim',
                        default=None)
    parser.add_argument('--bbox', 
                        help='specify bounding box of the shapefiles manually (N-S-E-W, decimal lat/lon) [must be the same as the largest extyent]', 
                        nargs=4, 
                        type=float,
                        metavar=('N', 'S', 'E', 'W'),
                        dest='bbox',
                        default=None)
    parser.add_argument('--foreground-color', 
                        help=f'specifiy color of the foreground of the image (default : {DEFAULT_FOREGROUND_COLOR})', 
                        nargs=1, 
                        type=str,
                        metavar=('color'),
                        dest='inpt_foreground_color',
                        default=None)
    parser.add_argument('--background-color', 
                        help=f'specifiy color of the background of the image (default : {DEFAULT_BACKGROUND_COLOR})', 
                        nargs=1, 
                        type=str,
                        metavar=('color'),
                        dest='inpt_background_color',
                        default=None)
    parser.add_argument('--perimeter-only', 
                        help='output image is just a perimeter rather than a filled shape', 
                        action='store_true',
                        dest='only_perim',
                        default=False)

    args = parser.parse_args()

    if args.input_file_directory[-1:] != "/":
        input_dir = args.input_file_directory + "/"
    else:
        input_dir = args.input_file_directory

    print(f"Reading files from '{input_dir}'")

    if args.save_directory[-1:] != "/":
        save_dir = args.save_directory + "/"
    else:
        save_dir = args.save_directory

    print(f"Saving to '{save_dir}'")

    if args.bbox:
        bounds_eswn = [args.bbox[2], args.bbox[1], args.bbox[3], args.bbox[0]]
    elif args.largest_perim:
        bounds_eswn = get_shape_bounds(args.largest_perim)
    else:
        raise ValueError("Total bounds of the perimeter must be specified (--last-shapefile or --bbox)")

    if args.inpt_background_color:
        bck_color = args.inpt_background_color[0]
    else:
        bck_color = DEFAULT_BACKGROUND_COLOR

    if args.inpt_foreground_color:
        frg_color = args.inpt_foreground_color[0]
    else:
        frg_color = DEFAULT_FOREGROUND_COLOR

    input_files = sorted(glob.glob(f'{input_dir}*.shp'))

    if not input_files:
        raise FileNotFoundError("No input files found in specified directory.")
    else:
        num_input_files = len(input_files)
        print(f"Successfully found {num_input_files} files.")

    with tqdm(miniters=0, total=num_input_files, desc=f'Creating images from {num_input_files} data...', ascii=" ✎✏✐█") as progress:
        for input_file_path in input_files:
            input_shapes = gpd.read_file(input_file_path)
            output_file_name = input_file_path[input_file_path.rfind('/')+1:].replace('.shp', '')

            create_image_from_poly(input_shapes,
                                   args.save_directory,
                                   output_file_name,
                                   bounds_eswn,
                                   bck_color,
                                   frg_color,
                                   False if args.only_perim else True)
            
            progress.update()

    print("Done!")
