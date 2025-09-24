#!/bin/bash
source /Users/ryanpurciel/anaconda3/bin/activate /opt/anaconda3/envs/wrfsfire

echo "Images Test"
python3 "/Users/ryanpurciel/Development/nvfirelab/wrf-toa-grid/ShapefileToImage.py" \
	'/Users/ryanpurciel/Development/nvfirelab/wrf-toa-grid/SHAPE_FILES_LAHAINA' \
	'/Users/ryanpurciel/Development/nvfirelab/wrf-toa-grid/shapefile-polys' \
	--last-shapefile '/Users/ryanpurciel/Development/nvfirelab/wrf-toa-grid/SHAPE_FILES_LAHAINA/930.shp'
