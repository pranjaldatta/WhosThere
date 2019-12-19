conda env create -f env.yaml
eval "$(conda shell.bash hook)"
conda activate facenet
TARGDIR="$CONDA_PREFIX/lib/python3.7/site-packages"
BASEDIR=$(readlink -f "$0")
BASEDIR=$(dirname $BASEDIR)
cp -r $BASEDIR $TARGDIR
