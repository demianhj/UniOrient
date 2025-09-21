## env: conda activate data_anno && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH



## for dino 3d lifting
# from helper.prepare_data_custom_v1 import run
from helper.prepare_data_custom_v1_batch import run

## check dino 3d feature pc
# from helper.prepare_data_check import run
## orient data and save
# from helper.prepare_data_orient import run
run()

# python test_v1.py