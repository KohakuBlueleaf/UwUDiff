# UwUDiffusion

Codebase/framework for training Diffusion Model. (Not a "trainer")

## UwU-series

* UwUDiff (You are here): for Diffusion Model
* UwULLM (WIP): for Auto Regressive Model

## Usage

To running demo training, you should do these steps for setup the env.

```bash
git clone https://github.com/KohakuBlueleaf/UwUDiff
cd UwUDiff
python -m pip install -e .
python ./scripts/test_train.py ./config/train/demo_training_lycoris.yaml
```