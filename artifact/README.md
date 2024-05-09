# NetConfEval: Artifact Evaluation

This folder contains the scripts for the CoNEXT 2024 Artifact Evaluation.

## Run the Benchmark

To run the whole benchmark and replicate all the experiments to produce all the figures in the paper, run the following:

1. Export your [OPENAI_API_KEY](https://platform.openai.com/api-keys) in the environment:
```bash
export OPENAI_API_KEY="YOUR OPENAI KEY"
```

2. Create a `venv` and install the `requirements.txt`:
``` bash
virtualenv venv 
source venv/bin/activate
pip install -r requirements.txt
```

3. To run experiments with Huggingface models, install the additional packages and login with your Huggingface account with your [Access Token](https://huggingface.co/settings/tokens).
```bash
pip install -r requirements-hf.txt
huggingface-cli login
```

4. Run the `1_run_all.sh` script, specifying the number of runs, for example:
```bash
./1_run_all.sh -r 1
```

Note that the process requires: (i) OpenAI credits to run GPT-based experiments, and (ii) a local GPU to run HuggingFace models.

## Plot the Figures

Once the experiments are all completed, simply run the plotter script:
```bash
./2_plot_all.sh
```

The script will produce all the figures of the paper.