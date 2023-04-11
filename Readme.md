# Code & Data for Submission 792
This is an implementation of BootCF, and the work is submitted to KDD2023.
# Environment
- Python 3.6.12
- PyTorch 1.6.0
- NumPy 1.19.1
- tqdm 4.51.0
# Dataset
The datasets are sampled and placed in data/ directory.
- **NYTaxi** is data from first 5 days of NYTaxi.

# Training
- Train the bootstrapping framework: **python train_boot.py** This will generate node representations for the whole dataset and save them under the directory output/bootstrap/results.
- Train the downstream task:
    - Train model for OD/inflow/outflow prediction: **python train_flow.py --task=od/i/o**
    - Train model for travel time estimation: **python train_TTE.py**
- If there are extra parameters on suffix field in the first stage, the downstream need to declare the path to representation by emd_path field:
    - Train model for OD/inflow/outflow prediction: **python train_flow.py --task=od/i/o --emd_path=/path/to/node/representations**
    - Train model for travel time estimation: **python train_TTE.py --emd_path=/path/to/node/representations**

More hyperparameters can be viewed in the source code and adjusted with the running demand.