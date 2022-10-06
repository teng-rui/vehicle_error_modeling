venv set up

conda create -n venvname
conda activate venvname
conda install --file requirements.txt
# install them separately: 
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge pyfmi (didn't work)