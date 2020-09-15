## Example json files for hyper-parameter runs
There are two ways to use `run_experiment_graphnet.py`:
* Pass the parameters explicitly
* pass the paths to two json files.

here an example of these files are given. One is for training parameters and one is for model parameters.

```
  python3 run_experiment_graphnet.py ---from-model-json example_json_hppruns/e1f68_model.json --training-options-json examples_json_hppruns/training_options.json

```
The json with the model hyper parameters can be created using `get_params_from_hash.py <hash>`

