from dataclasses import dataclass, field
from typing import Iterable, Optional, Mapping, Sequence, Tuple, List, Dict, Set
import abc
import math
import json
import collections
import random
import logging
import itertools
import importlib

import torch
import datasets

from utils import set_seed

DATASETS = ["agnews", "atis-intent", "cola", "cr", "dbpedia",
            "dbpedia-content", "imdb", "mpqa", "mr", "snips-intent", "sst2",
            "sst5", "subj", "trec6", "yahoo-answers", "yelp", "rt20"]
DATASET_VALID_SIZE = {
    "agnews": 2000,
    "dbpedia-content": 2000,
    "yahoo-answers": 5000,
    "imdb": 2000
}
DATASET_METATYPES = {
    "sst2": {
        "text_type": "movie review",
        "label_type": "sentiment"
    },
    "agnews": {
        "text_type": "news headline",
        "label_type": "classification"
    },
    "dbpedia-content": {
        "text_type": "description",
        "label_type": "classification"
    },
    "yahoo-answers": {
        "text_type": "question-answer pair",
        "label_type": "question type"
    },
    "imdb": {
        "text_type": "movie review",
        "label_type": "sentiment"
    },
    "cola": {
        "text_type": "text",
        "label_type": "grammar"
    },
    "snips-intent": {
        "text_type": "user request",
        "label_type": "intent type"
    },
    "cr": {
        "text_type": "customer review",
        "label_type": "sentiment"
    },
    "atis-intent": {
        "text_type": "user request",
        "label_type": "intent type"
    },
    "yelp": {
        "text_type": "place review",
        "label_type": "sentiment"
    },
    "subj": {
        "text_type": "text",
        "label_type": "objective"
    },
    "trec6": {
        "text_type": "question",
        "label_type": "type"
    },
    "mpqa": {
        "text_type": "phrase",
        "label_type": "sentiment"
    },
    "rt20": {
        "text_type": "movie review",
        "label_type": "sentiment"
    }
}
DATASET_LABEL_MAP = {
    # Note: verbalizing label must correspond to single
    # labels in GPT-3 vocabulary
    "agnews": {
        "Sci/Tech": "technology",
        "World": "world",
        "Sports": "sports",
        "Business": "business"
    },
    "imdb": {
        "pos": "positive",
        "neg": "negative"
    },
    "cola": {
        "0": "incorrect",
        "1": "correct"
    },
    "snips-intent": {
        "RateBook": "rate",
        "BookRestaurant": "book",
        "GetWeather": "weather",
        "SearchCreativeWork": "creative",
        "SearchScreeningEvent": "screening",
        "PlayMusic": "music",
        "AddToPlaylist": "playlist"
    },
    "cr": {
        "pos": "positive",
        "neg": "negative"
    },
    "atis-intent": {
        "atis_flight": "flight",
        "atis_quantity": "quantity",
        "atis_city": "city",
        "atis_ground_service#atis_ground_fare": "ground",
        "atis_ground_service": "ground",
        "atis_abbreviation": "abbreviation",
        "atis_airfare": "cost",
        "atis_flight#atis_airfare": "flight",
        "atis_flight_time": "time",
    },
    "yelp": {
        "positive": "positive",
        "neutral": "neutral",
        "negative": "negative",
        "very positive": "positive",
        "very negative": "negative"
    },
    "trec6": {
        "ABBR": "abbreviation",
        "LOC": "location",
        "DESC": "description",
        "NUM": "numeric",
        "ENTY": "entity",
        "HUM": "human"
    },
    "mpqa": {
        "pos": "positive",
        "neg": "negative"
    },
    "subj": {
        "objective": "yes",
        "subjective": "no"
    },
    "rt20": {
        "positive": "positive",
        "negative": "negative"
    }
}


@dataclass
class Example:
    text: str
    label: str
    probs: Optional[Mapping[str, float]] = None
    info: Optional[Mapping[str, str]] = None

    @property
    def is_soft(self):
        return self.probs is not None

    def to_dict(self):
        ret = {
            "text": self.text,
            "label": self.label
        }
        if self.probs is not None:
            ret["probs"] = self.probs
        if self.info is not None:
            ret["info"] = self.info
        return ret

    @classmethod
    def from_dict(cls, data):
        return Example(
            text=data["text"],
            label=data["label"],
            probs=data.get("probs"),
            info=data.get("info")
        )

    def __str__(self):
        if self.probs is None:
            label_str = f"({self.label})"
        else:
            probs_str = " / ".join(itertools.starmap("{}: {:.1%}".format,
                                                     self.probs.items()))
            label_str = "(" + probs_str + ")"
        return f"{self.text} {label_str}"


class Dataset(abc.ABC, Sequence):

    @abc.abstractmethod
    def get(self, idx):
        raise NotImplementedError

    def get_multi(self, idxs):
        return list(map(self.get, idxs))

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get(item)
        elif isinstance(item, Sequence):
            return self.get_multi(item)
        else:
            raise TypeError(f"unsupported index type: {type(item)}")


@dataclass
class Augmentation:
    fake: Example
    real: Optional[Sequence[Example]] = None

    def to_dict(self):
        return {
            "real": [ex.to_dict() for ex in self.real] if self.real else None,
            "fake": self.fake.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            real=(list(map(Example.from_dict, data["real"]))
                  if "real" in data else None),
            fake=Example.from_dict(data["fake"])
        )


@dataclass
class AugmentedDataset(Dataset):
    data: Sequence[Example]
    augmenter: "Augmenter"
    multiplier: int = 1
    reuse: bool = False
    save_path: str = None  # jsonlines
    num_classes: int = 5
    num_examples: int = 2
    sampling_strategy: str = "uniform"
    _class_map: Mapping[str, list] = field(init=False, hash=False, repr=False)
    _cache: List[Optional[Augmentation]] = \
        field(init=False, hash=False, repr=False)  # For caching augmentations
    _cache_filled: int = 0
    _pairwise_score: Dict[Tuple[int, int], float] = \
        field(init=False, hash=False, repr=False)

    def __post_init__(self):
        self._cache = [None] * (len(self.data) * self.multiplier)

        if self.sampling_strategy == "uniform":
            self._sample_examples = self._sample_examples_uniform
        elif self.sampling_strategy == "class-balanced":
            self._sample_examples = self._sample_examples_balanced
        elif self.sampling_strategy in {"furthest", "closest"}:
            self._sample_examples = self._sample_examples_score
            self._pairwise_score = dict()

            scorer = importlib.import_module("bleurt.score").BleurtScorer()
            for i, d1 in enumerate(self.data):
                for j, d2 in enumerate(self.data):
                    score = scorer.score(references=[d1.text],
                                         candidates=[d2.text])[0]
                    if self.sampling_strategy == "furthest":
                        score = -score
                    self._pairwise_score[(i, j)] = score

            self._pairwise_score = \
                {k: math.exp(v) for k, v in self._pairwise_score.items()}
            total_score = sum(self._pairwise_score.values())
            self._pairwise_score = \
                {k: v / total_score for k, v in self._pairwise_score.items()}
        else:
            raise ValueError(f"unsupported strategy: {self.sampling_strategy}")

        self._class_map = collections.defaultdict(list)
        for ex in self.data:
            self._class_map[ex.label].append(ex)

    def _save_augs(self, augs: Sequence[Augmentation]):
        if self.save_path is None:
            return

        with open(self.save_path, "a") as f:
            for aug in augs:
                f.write(json.dumps(aug.to_dict(), ensure_ascii=False) + "\n")

    def _sample_examples_wrapper(self):
        if not self.num_examples:
            return []
        if self.num_examples == len(self.data) and \
                self.sampling_strategy != "paraphrase":
            return self.data
        if self.num_examples > len(self.data):
            logging.warning(f"number of examples needed exceeds "
                            f"the data size:"
                            f" {self.num_examples} > {len(self.data)}")
        return self._sample_examples(min(len(self.data), self.num_examples))

    def _sample_examples_score(self, num_examples):
        if num_examples == 1:
            return random.choice(self.data)
        anchor_idx = random.choice(range(len(self.data)))
        anchor = self.data[anchor_idx]
        pairs, probs = \
            zip(*(((anchor, d), self._pairwise_score[(anchor_idx, d)])
                  for d in range(len(self.data)) if anchor_idx != d))
        sampled_pairs = random.choices(pairs, weights=probs, k=num_examples)
        return [anchor] + [self.data[b] for a, b in sampled_pairs]

    def _sample_examples_uniform(self, num_examples):
        return random.sample(self.data, num_examples)

    def _sample_examples_balanced(self, num_examples):
        num_classes = len(self._class_map)
        if num_examples % num_classes:
            raise ValueError(f"num_examples ({num_examples}) "
                             f"is not a multiple of number of "
                             f"classes ({num_classes})")

        num_examples_per_class = num_examples // num_classes
        ret = []
        for cls, class_data in self._class_map.items():
            if num_examples_per_class > len(class_data):
                raise RuntimeError(
                    f"number of examples per class ({num_examples_per_class}) "
                    f"exceeds the available data in the class "
                    f"({cls}: {len(class_data)})")

            ret.extend(random.sample(class_data, num_examples_per_class))
        random.shuffle(ret)
        return ret

    def _augment(self, *examples):
        augs = self.augmenter(*examples)
        self._save_augs(augs)
        return augs

    def get(self, idx):
        return self.get_multi([idx])[0]

    def get_multi(self, idxs):
        real_idxs = list(filter(lambda x: x < len(self.data), idxs))
        aug_idxs = list(filter(lambda x: x >= len(self.data), idxs))
        reals = list(map(self.data.__getitem__, real_idxs))
        if not self.reuse:
            augs = self._augment(*(self._sample_examples_wrapper()
                                   for _ in aug_idxs))
        else:
            req_idxs = [idx for idx in aug_idxs
                        if self._cache[idx - len(self.data)] is None]
            req_augs = self._augment(*(self._sample_examples_wrapper()
                                       for _ in req_idxs))
            for idx, aug in zip(req_idxs, req_augs):
                if aug is not None:
                    self._cache_filled += 1
                self._cache[idx - len(self.data)] = aug
            augs = [self._cache[idx - len(self.data)] for idx in aug_idxs]
        ret = []
        for idx in idxs:
            ret.append(reals.pop(0) if idx < len(self.data)
                       else augs.pop(0).fake)

        if self.reuse and self.fill_rate < 1.0:
            logging.info(f"{self.augmenter.__class__.__name__} "
                         f"current fill rate: {self.fill_rate * 100:.5f}%")

        return ret

    def __len__(self):
        return len(self.data) * (self.multiplier + 1)

    @property
    def fill_rate(self):
        """Returns the ratio of augmentation examples filled in. Only
        valid when the flag for reusing examples (`reuse`) is turned on.

        When `reuse` is off, each augmentation example will be generated
        on the fly, thus fill rate doesn't make sense in this case."""
        if not self.reuse:
            raise RuntimeError("fill rate cannot be reported for `reuse`=False")

        return self._cache_filled / len(self._cache)


@dataclass
class SequenceDataset(Dataset):
    data: Sequence

    def get(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


@dataclass
class DataSplits:
    train: Dataset
    valid: Dataset
    test: Dataset

    def to_dict(self):
        return {
            "train": self.train,
            "valid": self.valid,
            "test": self.test
        }


def create_dataloader(data, **kwargs):
    return torch.utils.data.DataLoader(
        data, collate_fn=lambda x: x, **kwargs
    )


def load_data(name: str, default_valid_size: int = 4000, valid_seed=42,
              label_map: Mapping = None
              ) -> Tuple[Dict[str, List[Example]], Set[str], dict]:
    datadict = datasets.load_dataset(name)

    metatype = DATASET_METATYPES \
        .get(name, {"text_type": "text", "label_type": "label"})
    label_map = label_map or DATASET_LABEL_MAP.get(name)
    dataset = {k: transform(v, label_map) for k, v in datadict.items()}
    labels = set(datadict["train"].features["label"].names)
    if label_map is not None:
        labels = {label_map[label] for label in labels}

    if "validation" not in datadict.keys():
        logging.info(f"{name} dataset contains no "
                     f"pre-split validation set")
        num_classes = datadict["train"].features["label"].num_classes
        valid_size = DATASET_VALID_SIZE.get(name, default_valid_size)
        samples_per_class = round(valid_size / num_classes)

        set_seed(valid_seed, "sampling validation set")
        logging.info(f"sampling {samples_per_class} per class "
                     f"(total {num_classes} classes) for "
                     f"the validation set")
        valid, train = balanced_samples(dataset["train"], samples_per_class)
        dataset["train"] = train
        dataset["validation"] = valid

    return dataset, labels, metatype


DataSequence = Sequence[Example]


def balanced_samples(data: DataSequence, size: int) -> Sequence[DataSequence]:
    """
    Arguments:
        data: sequence of examples
        size: list of target number of samples per class
    Returns:
        list of samples and the rest
    """
    cache = collections.defaultdict(list)
    for d in data:
        cache[d.label].append(d)
    for l in cache.values():
        random.shuffle(l)

    sample_set = list()
    for label, examples in cache.items():
        if len(examples) < size:
            logging.warning(f"number of target samples ({size}) more than "
                            f"the number of examples ({len(examples)}) "
                            f"in the class ({label})")
        for _ in range(min(size, len(examples))):
            sample_set.append(examples.pop())

    return sample_set, sum(cache.values(), [])


def transform(dataset: datasets.Dataset, label_map=None) -> List[Example]:
    int2str = dataset.features["label"].int2str
    if label_map is not None:
        def create_label(x):
            return label_map[int2str(x)]
    else:
        create_label = int2str
    return [Example(d['text'].replace("…", "..."), create_label(d['label']))
            for d in dataset]
