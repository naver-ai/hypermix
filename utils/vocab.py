__all__ = ["Vocabulary", "VocabularyFactory"]

import copy

from dataclasses import dataclass, field
from typing import TypeVar, Mapping, Sequence, Iterable

T = TypeVar("T")


@dataclass
class Vocabulary:
    f2i: Mapping[T, int] = field(default_factory=dict)
    i2f: Mapping[int, T] = field(default_factory=dict)

    def add(self, w: T, ignore_err=True) -> int:
        if w in self.f2i:
            if not ignore_err:
                raise ValueError(f"'{w}' already exists")
            return self.f2i[w]
        idx = len(self.f2i)
        self.f2i[w] = idx
        self.i2f[idx] = w
        return self.f2i[w]

    def clone(self):
        return Vocabulary(
            f2i=copy.copy(self.f2i),
            i2f=copy.copy(self.i2f)
        )

    def __iter__(self):
        return iter(self.f2i)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.i2f[item]
        else:
            return self.f2i[item]

    def __contains__(self, item):
        return item in self.f2i or item in self.i2f

    def __len__(self):
        return len(self.f2i)

    def _check_conflict(self, other):
        assert isinstance(other, Vocabulary)
        words = set(self.f2i) & set(other.f2i)
        for w in words:
            if self.f2i[w] != other.f2i[w]:
                raise ValueError(
                    f"subtraction failed because the two vocabs contain "
                    f"different indices for the same "
                    f"words: ({w}) {self.f2i[w]} != {other.f2i[w]}"
                )

    def __sub__(self, other):
        if not isinstance(other, Vocabulary):
            raise TypeError(f"not a valid type for subtraction: {other}")
        words = set(self.f2i) & set(other.f2i)
        self._check_conflict(other)
        return Vocabulary(
            f2i={w: i for w, i in self.f2i.items() if w not in words},
            i2f={i: w for w, i in self.f2i.items() if w not in words}
        )

    def __add__(self, other):
        if not isinstance(other, Vocabulary):
            raise TypeError(f"not a valid type for subtraction: {other}")
        self._check_conflict(other)
        return Vocabulary(
            f2i=dict(**self.f2i, **other.f2i),
            i2f=dict(**self.i2f, **other.i2f)
        )


@dataclass
class VocabularyFactory:
    max_size: int = None
    reserved: Sequence[T] = field(default_factory=tuple)
    counter: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        if self.max_size is not None:
            assert self.max_size > 0, \
                f"the size of the vocabulary must be larger than 0: " \
                f"{self.max_size} <= 0"

    def update(self, words: Iterable[T]):
        for w in words:
            if w not in self.counter:
                self.counter[w] = 0
            self.counter[w] += 1

    def get_vocab(self, vocab: Vocabulary = None):
        """Create (or update an existing) vocabulary object from the
        accumulated word counter data.

        Arguments:
            vocab (Vocabulary): (optional) an existing vocabulary object. If
                not provided, a new object will be returned.

        Returns:
            vocab (Vocabulary): a vocabulary object initialized with
                accumulated words and reserved words according to constraints
                specified by the fields
        """
        if vocab is None:
            vocab = Vocabulary()
        for w in self.reserved:
            vocab.add(w)
        if self.max_size is not None:
            cutoff = max(0, self.max_size - len(vocab))
            if not cutoff:
                return vocab
            counts = list(sorted(self.counter.items(), key=lambda x: x[1]))
            words, _ = zip(*counts[-cutoff:])
        else:
            words = self.counter.keys()
        for w in words:
            vocab.add(w)
        return vocab
