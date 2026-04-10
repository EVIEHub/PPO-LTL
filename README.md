# PPO-LTL: Safe RL with Linear Temporal Logic Constraints

**Internal Reference Repository for Collaborators**

This repository contains the research codebase for safe reinforcement learning experiments using PPO with LTL (Linear Temporal Logic) constraints. Includes implementations for CARLA autonomous driving scenarios and Zones grid-world environments.

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU
- CARLA Simulator 0.9.13+ (for CARLA experiments)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For CARLA experiments
pip install -r CRL/requirements.txt

# For Zones experiments
pip install -r deep-ltl/requirements.txt
```

## Repository Structure

```
├── CRL/                      # CARLA experiments
│   ├── carla_env/           # CARLA environment wrappers
│   ├── clip/                # VLM (CLIP) reward implementation
│   ├── shielding/           # Shield/masking wrapper
│   ├── ppo_lag.py           # PPO-Lagrangian implementation
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   └── config.py            # Configurations
│
├── deep-ltl/                # Zones environment experiments
│   ├── src/                 # Core source code
│   │   ├── envs/           # Environment definitions
│   │   ├── ltl/            # LTL automaton modules
│   │   ├── model/          # Policy networks
│   │   └── train/          # Training utilities
│   ├── CRL/
│   │   ├── ldba_wrapper.py      # LTL automaton wrapper
│   │   ├── ppo_lag.py           # PPO-Lagrangian
│   │   ├── train_ppo.py         # Training script
│   │   └── zone_shield_wrapper.py
│   └── rabinizer4/         # LTL to automaton tool
│
├── carla_env/               # Shared CARLA env (alternative location)
├── clip/                    # Shared CLIP module (alternative location)
├── pls/                     # Programmatic Labeling System tools
│
├── config.py                # Global configuration
├── train.py                 # Main training script
├── eval.py                  # Main evaluation script
└── requirements.txt         # Dependencies
```

## Usage

### CARLA Experiments

**Training:**
```bash
cd CRL

# PPO-LTL
python train.py --config tirl_ppo --seed 42

# Vanilla PPO
python train.py --config vanilla_ppo --seed 42

# PPO with shielding
python train.py --config ppo_shield --seed 42
```


### Zones Experiments

**Training:**
```bash
cd deep-ltl

# PPO-LTL on Zones
python run_zones.py --algo ppoltl --seed 100


## Key Components

### 1. LTL Monitoring
- **`deep-ltl/CRL/ldba_wrapper.py`**: Core LTL automaton wrapper
- **`deep-ltl/src/ltl/`**: LTL specification and automaton modules
- **`rabinizer4/`**: LTL to automaton conversion tool

### 2. Lagrangian Optimization
- **`CRL/ppo_lag.py`**: PPO-Lag for CARLA
- **`deep-ltl/CRL/ppo_lag.py`**: PPO-Lag for Zones

### 3. Shielding/Masking
- **`CRL/shielding/shield_wrapper.py`**: Reactive action masking
- **`deep-ltl/CRL/zone_shield_wrapper.py`**: Zones shield

## Configuration

Main config files:
- **`CRL/config.py`**: CARLA experiments
- **`deep-ltl/src/config/`**: Zones experiments
- **`config.py`**: Global parameters

Key hyperparameters:
- Learning rate: `1e-4`
- Lagrangian LR: `0.01 - 0.05`
- Cost limit: `0.05 - 0.10`
- GAE lambda: `0.95`
- Discount: `0.98`

## Experiments

### CARLA
- Urban driving scenarios
- Traffic light compliance
- Collision avoidance
- LTL: "Always safe AND eventually reach goal"

### Zones
- Grid-world navigation
- Multi-objective temporal tasks
- LTL: "Visit blue, then green, then yellow"

## Methods Comparison

| Method | Description |
|--------|-------------|
| **PPO-LTL** | PPO + LTL monitoring + Lagrangian |
| **PPO-Lag** | PPO + Lagrangian (no LTL) |
| **PPO-Shield** | PPO + reactive masking |



## Dependencies

Core libraries:
- `torch >= 1.12.0`
- `stable-baselines3 >= 1.6.0`
- `gymnasium >= 0.26.0`
- `carla >= 0.9.13`
- `transformers` (CLIP)
- `opencv-python`, `numpy`, `pandas`

## Notes

- Research codebase for internal reference
- Model weights and results not included
- Requires GPU for CARLA experiments
- See individual directories for detailed docs

## Contact

For questions, open an issue or contact maintainers directly.
