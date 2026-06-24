


## Train from config

```bash
python train.py --config config/config.yaml
```

## Override values from the command line

```bash
python train.py \
  --config config/config.yaml \
  --condition blocking \
  --episodes 5000 \
  --random-start \
  --start-margin 15 \
  --set vehicle.action_step=10
```

## Run a saved policy

```bash
python run_policy.py \
  --policy outputs/dqn/policies/best_policy.pt \
  --render \
  --pixels-per-unit 2
```

## Multi-run launcher

Uses the `multi_run` section in the config:

```bash
python run_many.py --launcher-config config/config.yaml
```

Or specify runs manually:

```bash
python run_many.py \
  --launcher-config config/config.yaml \
  --configs config/config.yaml \
  --conditions blocking crossing orbit \
  --seeds 7 8 9 \
  --episodes 3000 \
  --random-start \
  --sweep vehicle.action_step=10,20
```


