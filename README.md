# Setup
```
git clone --recurse-submodules https://github.com/4ku/robot_manipulation
conda create -n jaxrl python=3.10
conda activate jaxrl
pip install -e . 
cd bridge_data_v2
pip install -e . 
pip install -r requirements.txt
pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install pytz
```


# Run
```
python train.py  --config config.py:gc_bc     --bridgedata_config bridge_data_v2/experiments/configs/data_config.py:all     --name sac_rnd
```