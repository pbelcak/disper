# DisPer -- Disentangling Permutations
The code repository for the paper "End-to-End Neural Permutation Program Synthesis".

## Overview

### Requirements
Check out `requirements.txt`, or directly use `pip` to install them.
```
matplotlib==3.5.1
numpy==1.22.3
torch==1.11.0
wandb==0.12.18
```

### Structure
All important code files are on the root level of the repository.

 - `main.py` handles command-line arguments and starts individual jobs/creates sweep agents (sweeping hyperparameter tuning is done through WandB)
 - `experiment.py` is where the experiment gets initialized and run, epoch by epoch
 - `curriculum.py` contains the definition of training losses and training loops (batch level)
 - `evaluation.py` contains the procedures for computing evaluation metrics
 - `permutation_model.py` sets out the neural model architecture

### Run Options
```
usage: main.py [-h] [-j JOB_ID] [-o OUTPUT_PATH] [-t TASK] [-df DATA_FORM] [-ddp DATA_DIRECTORY_PATH] [-da {load,generate,generate_and_save}] [-f FOCUS] [-m {single,sweep}] [--sweep-id SWEEP_ID] [--sweep-runs SWEEP_RUNS] [--verbosity VERBOSITY] [--wandbosity WANDBOSITY]
               [--plotlibsity PLOTLIBSITY]

optional arguments:
  -h, --help            show this help message and exit
  -j JOB_ID, --job-id JOB_ID
                        The job id (and the name of the wandb group)
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        The directory which will contain job sub-directories containing the checkpoints of best models
  -t TASK, --task TASK  The task to perform
  -df DATA_FORM, --data-form DATA_FORM
                        The data form to use when creating the training (default: unique_n)
  -ddp DATA_DIRECTORY_PATH, --data-directory-path DATA_DIRECTORY_PATH
                        The path to the directory containing the data to use (default: ./data)
  -da {load,generate,generate_and_save}, --data-action {load,generate,generate_and_save}
                        The action to perform when preparing the data (choices: load, generate, generate_and_save; default: load)
  -f FOCUS, --focus FOCUS
                        The rolling ID of the group to focus at (default 0, means all groups being loaded)
  -m {single,sweep}, --mode {single,sweep}
                        Choose whether to do a single evaluation run or whether to start a sweep agent (default: single)
  --sweep-id SWEEP_ID   The id of the sweep to connect to (usually a string rather than a number)
  --sweep-runs SWEEP_RUNS
                        The number of sweep runs to do in this job (default: 1)
  --verbosity VERBOSITY
                        The terminal output verbosity level (0 is min, 2 is max, default: 2)
  --wandbosity WANDBOSITY
                        The level of verbosity for wandb (0 is min, 2 is max, default: 2)
  --plotlibsity PLOTLIBSITY
                        The level of verbosity for matplotlib (0 is min, 1 is max, default: 1)
```

### So, how do I run this thing?
Say that you want to reconstruct all groups of order less than `10` in a single job, saving the checkpoints of the best models to `./bests`.
Then, in the root directory of the repository, just run
```
python main.py -o ./bests -t full_recovery
```
The plots of evolving atoms/primitives can be found under `./plots`.

More details about various data representations and tasks can be found in the paper.

## Feedback, Questions, Issues, and Pull Requests
If you have any feedback to offer, a question about or an issue with the implementation, or a pull request to make, just use GitHub Issues functionality, or hit me up at <belcak@ethz.ch>.
