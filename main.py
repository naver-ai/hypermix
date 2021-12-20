import gc
import sys
import json
import shlex
import argparse
import pprint
import pathlib
import logging
import random
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
import transformers

from utils import set_seed, generate_seeds
from models import AbstractClassifier, TransformersClassifier, TMixClassifier
from dataset import DataSplits, SequenceDataset, AugmentedDataset, load_data, \
    balanced_samples
from augment import GPT3MixAugmenter
from openai_utils import resolve_api_key


@dataclass
class Runner:
    model: AbstractClassifier
    master_seed: Optional[int] = None

    def run_single(self, data_splits: DataSplits):
        gc.collect()
        torch.cuda.empty_cache()
        self.model.reset()

        logging.info("running single trial...")
        summary = self.model.fit(data_splits.train, data_splits.valid)
        if summary:
            logging.info(f"training summary: {json.dumps(summary, indent=2)}")

        pred = self.model.predict([ex.text for ex in data_splits.test])
        acc = np.mean([p == ex.label for p, ex in zip(pred, data_splits.test)])
        logging.info(f"final test acc: {acc}")

        return acc

    def run_multiple(self, data_splits: DataSplits, n, prepare_fn=None):
        seed_generator = generate_seeds(self.master_seed, "running experiments")
        seeds = list(seed for _, seed in zip(range(n), seed_generator))
        logging.info(f"generated seeds: {seeds}")

        accs = []
        for run_idx, seed in enumerate(seeds):
            if prepare_fn is None:
                prepared_data_splits = data_splits
            else:
                prepared_data_splits = prepare_fn(run_idx, data_splits)

            set_seed(seed, "running a single trial")
            results = self.run_single(prepared_data_splits)
            accs.append(results)

        return accs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", nargs="+", required=True, 
                        choices=["gpt3mix/rt20", "gpt3mix/sst2"],
                        default=[])

    group = parser.add_argument_group("Experiment Options")
    group.add_argument("--num-trials", type=int, default=1)
    group.add_argument("--master-exp-seed", type=int, default=42)
    group.add_argument("--master-data-seed", type=int, default=42)
    group.add_argument("--data-seeds", type=int, default=[],
                       action="append",
                       help="Set data seeds for sub-sampling. "
                            "If --num-trials > the number of data seeds, then "
                            "the data seeds will be used in "
                            "the round-robin fashion.")
    group.add_argument("--progress", default=False, action="store_true")
    group.add_argument("--save-dir", default="/tmp/out")

    group = parser.add_argument_group("Data Options")
    group.add_argument("--train-subsample",
                       help="Subsample is given as "
                            "number of samples per class (e.g. 3s) or "
                            "as a class-balanced ratio (e.g. 0.1f).")
    group.add_argument("--valid-subsample",
                       help="Subsample is given as "
                            "number of samples per class (e.g. 3s) or "
                            "as a class-balanced ratio (e.g. 0.1f).")
    group.add_argument("--test-subsample",
                       help="Subsample is given as "
                            "number of samples per class (e.g. 3s) or "
                            "as a class-balanced ratio (e.g. 0.1f).")
    group.add_argument("--default-metatype", type=int, default=0,
                       help="Use default metatypes ('text' and 'label')")
    group.add_argument("--text-type-override")
    group.add_argument("--label-type-override")
    group.add_argument("--label-map-override")

    group = parser.add_argument_group("OpenAI GPT-3 Options")
    group.add_argument("--api-key",
                       help="WARN: Save the api-key as 'openai-key' "
                            "in the working directory instead.")
    group.add_argument("--gpt3-engine", default="ada",
                       choices=("ada", "babbage", "curie", "davinci"))
    group.add_argument("--gpt3-batch-size", type=int, default=20)
    group.add_argument("--gpt3-num-examples", type=int, default=10)
    group.add_argument("--gpt3-frequency-penalty", type=float, default=0.01)
    group.add_argument("--gpt3-max-retries", type=int, default=10)

    group = parser.add_argument_group("Classifier Options")
    group.add_argument("--classifier", default="transformers",
                       choices=["transformers", "tmix"])
    group.add_argument("--model-name", default="distilbert-base-uncased")
    group.add_argument("--batch-size", type=int, default=32)
    group.add_argument("--patience", type=int, default=10)
    group.add_argument("--optimizer", default="AdamW")
    group.add_argument("--lr", type=float, default=5e-5)

    group = parser.add_argument_group("Augmentation Options")
    group.add_argument("--augmenter", default="none",
                       choices=("none", "gpt3-mix"))
    group.add_argument("--reuse", type=int, default=1,
                       help="Whether to reuse generated synthetic augmentation")
    group.add_argument("--multiplier", type=int, default=1,
                       help="Ratio of real-to-synthetic data.")
    group.add_argument("--num-examples", type=int, default=2,
                       help="Number of examples to use for generating "
                            "augmentation sample.")
    group.add_argument("--num-classes", type=int, default=2,
                       help="Number of classes to use for generating "
                            "augmentation sample.")
    group.add_argument("--example-sampling", default="uniform",
                       choices=("uniform", "furthest", "closest",
                                "class-balanced"),
                       help="Example sampling strategy.")

    group = parser.add_argument_group("GPT3Mix Specific Options")
    group.add_argument("--gpt3-mix-max-tokens", type=int, default=100)
    group.add_argument("--gpt3-mix-soft", type=int, default=1)

    group = parser.add_argument_group("TMix Options")
    group.add_argument("--tmix-alpha", type=float, default=0.75)
    group.add_argument("--tmix-layers", type=int, nargs="+", default=[7, 9, 12])

    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = pathlib.Path(args.save_dir)
    if save_dir.exists():
        warnings.warn(f"saving directory {save_dir} already exists. "
                      f"overwriting...")
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(save_dir.joinpath("run.log")),
            logging.StreamHandler()
        ]
    )

    resolve_api_key(args)

    logging.info("Command-line Arguments:")
    logging.info(pprint.pformat(vars(args)))
    logging.info(f"Raw command-line arguments: "
                 f"{' '.join(map(shlex.quote, sys.argv))}")

    for dataset in args.datasets:
        logging.info(f"loading {dataset} dataset...")
        data, label_set, metatype = load_data(
            name=dataset,
            label_map=eval(args.label_map_override or "None")
        )

        if args.default_metatype:
            metatype = {"text_type": "text", "label_type": "label"}

        if args.text_type_override:
            metatype["text_type"] = args.text_type_override

        if args.label_type_override:
            metatype["label_type"] = args.label_type_override

        label_set = set(label_set)
        data_splits = DataSplits(
            train=SequenceDataset(data["train"]),
            valid=SequenceDataset(data["validation"]),
            test=SequenceDataset(data["test"])
        )
        if not args.data_seeds:
            data_seed_generator = generate_seeds(args.master_data_seed, "data")
            data_seeds = [seed for _, seed in zip(range(args.num_trials),
                                                  data_seed_generator)]
            logging.info(f"generated data seeds: {data_seeds}")
        else:
            data_seeds = list(args.data_seeds)

        logging.info(f"sample dataset instance: "
                     f"{random.choice(data_splits.train)}")
        logging.info(f"label set: {label_set}")

        def create_augmenter():
            if args.augmenter == "gpt3-mix":
                augmenter = GPT3MixAugmenter(
                    api_key=args.api_key,
                    label_set=label_set,
                    engine=args.gpt3_engine,
                    batch_size=args.gpt3_batch_size,
                    label_type=metatype["label_type"],
                    text_type=metatype["text_type"],
                    max_tokens=args.gpt3_mix_max_tokens,
                    frequency_penalty=args.gpt3_frequency_penalty,
                    max_retries=args.gpt3_max_retries,
                    soft_label=bool(args.gpt3_mix_soft),
                    ignore_error=True
                )
                augmenter.construct_prompt(
                    random.sample(data_splits.train, args.num_examples),
                    demo=True
                )
                return augmenter
            elif args.augmenter == "none":
                return
            else:
                raise NotImplementedError(
                    f"unsupported augmenter: {args.augmenter}")

        def prepare_datasplits(run_idx, data_splits):
            def _prepare_train_data(train_data):
                if args.augmenter == "none":
                    return train_data

                aug_save_dir = save_dir.joinpath("augmentations")
                aug_save_dir.mkdir(parents=True, exist_ok=True)

                return AugmentedDataset(
                    data=list(train_data),
                    augmenter=create_augmenter(),
                    multiplier=args.multiplier,
                    reuse=args.reuse,
                    save_path=aug_save_dir.joinpath(
                        f"run-{run_idx:03d}.jsonlines"),
                    num_examples=args.num_examples,
                    num_classes=args.num_classes,
                    sampling_strategy=args.example_sampling
                )

            data_seed = data_seeds.pop(0)
            data_seeds.append(data_seed)
            sub_splits = data_splits.to_dict()
            for split in ("train", "valid", "test"):
                subsample_spec = getattr(args, f"{split}_subsample")

                if subsample_spec is None:
                    continue

                set_seed(data_seed, f"subsampling {split}")

                if subsample_spec.endswith("f"):
                    ratio = float(subsample_spec[:-1])
                    size = max(1, int(round(len(getattr(data_splits, split)) *
                                            ratio / len(label_set))))
                elif subsample_spec.endswith("s"):
                    size = int(subsample_spec[:-1])
                else:
                    raise ValueError(f"unsupported subsample "
                                     f"specification format: {subsample_spec}")

                logging.info(f"subsampling {size} instances per class "
                             f"(total {len(label_set)} classes) "
                             f"from {split} set...")
                subsample, _ = balanced_samples(
                    data=getattr(data_splits, split),
                    size=size
                )
                sub_splits[split] = SequenceDataset(subsample)

            sub_splits["train"] = _prepare_train_data(sub_splits["train"])
            s = DataSplits(**sub_splits)
            return s

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.classifier == "transformers":
            model = TransformersClassifier(
                label_set=label_set,
                tok_cls=transformers.AutoTokenizer,
                model_cls=transformers.AutoModelForSequenceClassification,
                model_name=args.model_name,
                batch_size=args.batch_size,
                patience=args.patience,
                optimizer_cls=lambda x: getattr(torch.optim, args.optimizer)(
                    params=x, lr=args.lr, eps=1e-8),
                device=device
            ).to(device)
        elif args.classifier == "tmix":
            model = TMixClassifier(
                label_set=label_set,
                model_name=args.model_name,
                batch_size=args.batch_size,
                patience=args.patience,
                optimizer_cls=lambda x: getattr(torch.optim, args.optimizer)(
                    params=x, lr=args.lr, eps=1e-8),
                device=device,
                alpha=args.tmix_alpha,
                mix_layer_set=frozenset(args.tmix_layers)
            )
        else:
            raise ValueError(f"unrecognized classifier type: {args.classifier}")

        if isinstance(model, torch.nn.Module):
            num_params = sum(np.prod(p.size())
                             for p in model.parameters() if p.requires_grad)
            logging.info(f"number of model params: {num_params:,d}")

        runner = Runner(
            model=model,
            master_seed=args.master_exp_seed
        )
        accs = runner.run_multiple(data_splits, args.num_trials,
                                   prepare_fn=prepare_datasplits)

        accs_str = ", ".join(map("{:.4f}".format, accs))
        logging.info(f"all accuracies: {accs_str}")
        logging.info(f"mean: {np.mean(accs)}, std: {np.std(accs)}")


if __name__ == "__main__":
    main()
