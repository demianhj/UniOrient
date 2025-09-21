## env: conda activate diffTheta3D && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



## for dino 3d lifting
# from helper.prepare_data_custom import run
## check dino 3d feature pc
# from helper.prepare_data_check import run
## orient data and save
from helper.prepare_data_orient import run
run()