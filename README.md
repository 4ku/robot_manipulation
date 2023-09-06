# Robot Manipulation: Goal-conditioned Behavioral Cloning with SAC-RND

## Installation

Follow these steps to set up the project environment:

1. **Clone the Repository**
    ```bash
    git clone --recurse-submodules https://github.com/4ku/robot_manipulation
    ```
2. **Create a Conda Environment**
    ```bash
    conda create -n jaxrl python=3.10
    ```
3. **Activate the Conda Environment**
    ```bash
    conda activate jaxrl
    ```
4. **Install Dependencies**
    ```bash
    pip install -e .
    cd bridge_data_v2
    pip install -e . 
    pip install -r requirements.txt
    pip install --upgrade "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

## How to Run

Execute the following command to train the agent:

```bash
python train.py  --config config.py:gc_bc --bridgedata_config bridge_data_v2/experiments/configs/data_config.py:all --name sac_rnd
```

## Techniques Used

- **Goal-conditioned Behavioral Cloning**: This method allows the agent to learn from expert demonstrations and is designed to help the agent understand task-specific goals. You can find the implementation details in the [BridgeData V2 repository](https://github.com/rail-berkeley/bridge_data_v2).

- **SAC-RND (Soft Actor-Critic with Random Network Distillation)**: This algorithm combines state-of-the-art reinforcement learning with exploration techniques to improve learning efficiency.

## Results

For detailed experiment results of Goal-conditioned Behavioral Cloning, please refer to the [Experiment Section](https://github.com/4ku/bridge_data_v2#experiments-and-results).

## Troubleshooting

Currently, the agent has not been able to learn properly with the given techniques. This is an area under investigation.