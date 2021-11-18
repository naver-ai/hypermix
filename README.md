# HyperMix

Code for our paper [GPT3Mix](https://arxiv.org/abs/2104.08826) and conducting classification experiments using GPT-3 prompt-based data augmentation.

## Getting Started

### Installing Packages

The main depedencies can be installed via `pip install -r requirements.txt`.

### Usage

The main code is run through `main.py`. Check out `--help` for full list of commands.

    python main.py --help

The code will automatically use the first GPU device, if detected.

A typical command to run BERT-base 10 times on the 1% subsample set of the SST-2 dataset and computing the average
of all run is as follows.

    python main.py --datasets sst2 \
        --train-subsample 0.01f \
        --classifier transformers \
        --model-name bert-base-uncased \
        --num-trials 1 \
        --augmenter none \
        --save-dir out

The script will create a directory named `out` in the current working directory and save the script log 
as `out/run.log`. It will also save any augmentations created during the experiments (if any augmentation is enabled).

To test GPT3Mix, prepare an OpenAI API key as described at the bottom of this README file, then use the following command:

    python main.py --datasets sst2 \
        --train-subsample 0.01f \
        --classifier transformers \
        --model-name bert-base-uncased \
        --num-trials 1 \
        --augmenter gpt3-mix \
        --save-dir out


### Managing Seeds

In the command above, the script will automatically generate seeds for sampling data and optimizing models.
The seed used to generate each individual seed is called "master seed" and can be set using `--master-data-seed`
and `--master-exp-seed` options. As evident from the option names, they are responsible for sampling data and
optimizing a freshly initialized models respectively.

Sometimes, we need to manually set the seeds and not rely on automatically generated seeds from the master seeds.
Manually seeding can be achieved via `--data-seeds` option. If this option is given, the master data seed will
be ignored. We only support manualy data seeding for now.

### OpenAI Key

Store OpenAI API Key under the current working directory as a file named `openai-key`. 
When running the main script, it will automatically detect the api key.

API keys can be provided to the script by `--api-key` option (not recommended) or from a file named `openai-key` in the current working directory.

### Other Notes

At the moment we only support data augmentation leveraging OpenAI GPT-3 (GPT3Mix), but we will release an update that supports [HyperCLOVA](https://arxiv.org/abs/2109.04650) as soon as it becomes available to the public (HyperMix).

## Citation

To cite our code or work, please use the following bibtex:

	@inproceedings{yoo2021gpt3mix,
		title = "GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation",
		author = "Yoo, Kang Min  and
		  Park, Dongju  and
		  Kang, Jaewook  and
		  Lee, Sang-Woo  and
		  Park, Woomyoung",
		booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
		month = nov,
		year = "2021",
		publisher = "Association for Computational Linguistics",
		url = "https://aclanthology.org/2021.findings-emnlp.192",
		pages = "2225--2239",
	}

