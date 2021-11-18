__all__ = ["Augmenter", "GPT3MixAugmenter"]

import math
import random
import bisect
import logging
import warnings
import collections
from dataclasses import dataclass, field
from typing import Sequence, Tuple
from abc import ABC, abstractmethod

import openai
import regex
from regex.regex import Pattern

from dataset import Example, Augmentation
from utils import retry, MaxRetryError


def _retry_completion(max_retries=10, **kwargs):
    return retry(lambda: openai.Completion.create(**kwargs), max_retries,
                 desc="OpenAI completion request")


class Augmenter(ABC):
    # Define the range of examples needed by the augmenter (inclusive).
    num_examples_range = (0, 100)

    @abstractmethod
    def augment(self, *examples: Sequence[Example]) -> Sequence[Augmentation]:
        raise NotImplementedError

    def __call__(self, *examples: Sequence[Example]) -> Sequence[Augmentation]:
        for i, example in enumerate(examples):
            if not (self.num_examples_range[0] <= len(example) <=
                    self.num_examples_range[1]):
                raise RuntimeError(f"number of {i}-th examples is not within"
                                   f" the range: {self.num_examples_range}")
        return self.augment(*examples)


def _normalize(x):
    return {k: v / sum(x.values()) for k, v in x.items()}


def _label_enum_str(label_set, or_str="or"):
    labels = list(label.lower().capitalize() for label in label_set)
    if len(labels) == 1:
        label_enum_str = labels[0]
    else:
        label_enum_str = (", ".join(map("'{}'".format, labels[:-1])) +
                          f" {or_str} '{labels[-1]}'")
    return label_enum_str


def construct_mix_description(text_type, label_type, label_set):
    return (
        f"Each item in the following list contains a "
        f"{text_type.lower()} and the respective "
        f"{label_type.lower()}. "
        f"{label_type} is one of {_label_enum_str(label_set).lower()}."
    )


@dataclass
class GPT3MixAugmenter(Augmenter):
    api_key: str
    label_set: set
    soft_label: bool = False
    engine: str = "ada"
    batch_size: int = 20
    label_type: str = "label"
    text_type: str = "text"
    max_tokens: int = 100
    frequency_penalty: float = 0.02
    max_retries: int = 3  # completion retries
    ignore_error: bool = False
    num_examples_range: Tuple[int, int] = (1, 100)
    _pattern: Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Warn about lowercase normalization
        for label in self.label_set:
            if not label.islower():
                warnings.warn(f"Note that label ('{label}') will be "
                              f"normalized to lower case.")

        self.label_set = set(label.lower() for label in self.label_set)
        self._pattern = regex.compile(
            rf"(?r){self._text_type}: (.*)\({self._label_type}: (.*)\)")

    @property
    def _label_type(self):
        return self.label_type.lower().capitalize()

    @property
    def _text_type(self):
        return self.text_type.lower().capitalize()

    def construct_prompt(self, examples: Sequence[Example], demo=False):
        prompt = construct_mix_description(
            text_type=self._text_type,
            label_type=self._label_type,
            label_set=self.label_set
        ) + "\n"

        for example in examples or []:
            # Warn about lower case normalization
            if not example.label.islower():
                warnings.warn(f"Example label ('{example.label}') will be "
                              "normalized to lower case first then "
                              "capitalized.")
            label = example.label.strip().lower().capitalize()
            prompt += (f"\n{self._text_type}: {example.text.strip()} "
                       f"({self._label_type}: {label})")
        prompt += f"\n{self._text_type}:"

        if demo:
            logging.info("constructed prompt:")
            logging.info(prompt)
        return prompt

    def _parse_choice(self, choice) -> Example:
        text = (choice["text"] if self.soft_label else
                f"{self._text_type}: {choice['text'].lstrip()}")
        match = self._pattern.search(text)

        if match is None:
            raise RuntimeError(f"unexpected completion text - "
                               f"regex pattern not found: {text}")

        text, text_span = match.group(1).strip(), match.span(1)
        label, label_span = match.group(2).strip().lower(), match.span(2)

        if self.soft_label:
            logprobs = choice["logprobs"]
            token_sidx = bisect.bisect_left(logprobs["text_offset"],
                                            label_span[0])

            if token_sidx > len(logprobs["text_offset"]):
                raise RuntimeError(
                    "label span offset exceeds the total text length")

            top_logprobs = logprobs["top_logprobs"][token_sidx - 1]

            label_probs = collections.defaultdict(float)
            for k, v in top_logprobs.items():
                norm_k = k.strip().lower()
                if norm_k in self.label_set:
                    label_probs[norm_k] += math.exp(v)

            if not label_probs:
                raise RuntimeError(f"no label-related tokens found "
                                   f"in the prob table: {top_logprobs}")

            label_probs = _normalize(label_probs)
        else:
            label_probs = None

        # if hard label and label is not one of label vocab then discard the
        # example
        if not self.soft_label and label not in self.label_set:
            raise ValueError(f"generated label '{label}' is not valid: "
                             f"not one of {self.label_set}")

        example = Example(
            text=text,
            label=(max(label_probs, key=lambda x: label_probs[x])
                   if self.soft_label else label),
            probs=label_probs,
        )

        return example

    def _try_augment(self, examples, prompts):
        openai_kwargs = dict(
            engine=self.engine,
            prompt=prompts,
            echo=self.soft_label,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            stop="\n",
        )

        if self.soft_label:
            openai_kwargs["logprobs"] = 100

        resp = _retry_completion(**openai_kwargs)

        batch_examples = examples
        ret, completed_idx = [], []
        for i, (examples, choice) in \
                enumerate(zip(batch_examples, resp.choices)):
            try:
                new_example = retry(
                    lambda: self._parse_choice(choice),
                    max_retries=1,
                    desc="parsing response choice"
                )
                augmentation = Augmentation(
                    real=examples,
                    fake=new_example
                )
                ret.append(augmentation)
                completed_idx.append(i)
            except MaxRetryError as e:
                logging.error(f"Parsing OpenAI Completion responses "
                              f"failed; Retry msg: {e}")
        return ret, completed_idx

    def augment(self, *examples: Sequence[Example]) -> Sequence[Augmentation]:
        prompts = [self.construct_prompt(ex) for ex in examples]
        openai.api_key = self.api_key
        augmentations: Sequence[Augmentation] = [None] * len(examples)
        incomplete_idx = set(range(len(examples)))

        retry_count = 0
        while incomplete_idx:
            cur_idx = random.sample(list(incomplete_idx),
                                    min(self.batch_size, len(incomplete_idx)))
            augmentation, completed_idx = self._try_augment(
                examples=[examples[i] for i in cur_idx],
                prompts=[prompts[i] for i in cur_idx]
            )

            if augmentation:
                retry_count = 0
            else:
                retry_count += 1

            if retry_count > self.max_retries:
                logging.error(f"Trying to augment through OpenAI GPT-3 "
                              f"completion failed. Most likely due to inability"
                              f" to parse the completion text correctly. "
                              f"Check GPT-3 settings.")
                raise MaxRetryError(f"no augmentation succeeded "
                                    f"for {self.max_retries} times")

            for idx, aug in zip(completed_idx, augmentation):
                augmentations[cur_idx[idx]] = aug
                incomplete_idx.remove(cur_idx[idx])

        assert all(aug is not None for aug in augmentations)
        return augmentations
