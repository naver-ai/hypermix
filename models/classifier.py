__all__ = ["AbstractClassifier", "TransformersClassifier"]

import logging
import itertools
import random
from abc import ABC, abstractmethod
from typing import Sequence, Mapping

import torch
import torch.nn as nn
import transformers

from dataset import Example, Dataset, create_dataloader
from .cross_entropy import CrossEntropy


class AbstractClassifier(ABC):

    @abstractmethod
    def fit(self, train_data: Sequence[Example],
            valid_data: Sequence[Example], **kwargs) -> Mapping:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Sequence[str], **kwargs) -> Sequence[str]:
        raise NotImplementedError


class TransformersClassifier(AbstractClassifier, nn.Module):

    def __init__(self, label_set,
                 tok_cls=transformers.AutoTokenizer,
                 model_cls=transformers.AutoModelForSequenceClassification,
                 model_name="bert-base-uncased",
                 batch_size=32,
                 patience=5,
                 optimizer_cls=None,
                 device=torch.device("cpu")):
        super().__init__()
        self.label_set = label_set
        self.batch_size = batch_size
        self.patience = patience
        self.optimizer_cls = (optimizer_cls or
                              (lambda x: torch.optim.AdamW(x, 5e-5)))
        self.device = device
        self.tok_cls = tok_cls
        self.model_cls = model_cls
        self.model_name = model_name
        self.tokenizer = self.tok_cls.from_pretrained(model_name)
        self.model = (self.model_cls
                      .from_pretrained(model_name, num_labels=len(label_set)))
        self.model = self.model.to(self.device)
        self._model_backup = {k: v.detach().cpu().clone() for k, v in
                              self.model.state_dict().items()}
        self.label_vocab = {label: i for i, label in enumerate(label_set)}
        self.label_vocab.update(
            {i: label for label, i in self.label_vocab.items()})
        self.loss_fn = CrossEntropy(reduction="none")

    def create_label_tensor(self, batch: Sequence[Example]):
        labels = []
        for ex in batch:
            label_tensor = torch.zeros(len(self.label_set)).float()
            if ex.is_soft:
                for label, prob in ex.probs.items():
                    idx = self.label_vocab[label]
                    label_tensor[idx] = prob
            else:
                idx = self.label_vocab[ex.label]
                label_tensor[idx] = 1.0
            labels.append(label_tensor)
        return torch.stack(labels).float().to(self.device)

    def compute_forward(self, texts: Sequence[str]):
        x = self.tokenizer.batch_encode_plus(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return self.model(**{k: v.to(self.device) for k, v in x.items()})[0]

    def compute_loss(self, batch: Sequence[Example]):
        logit = self.compute_forward([ex.text for ex in batch])
        target = self.create_label_tensor(batch)
        return self.loss_fn(logit, target)

    def _predict(self, texts: Sequence[str]):
        logit = self.compute_forward(texts)
        return [self.label_vocab[pred.item()] for pred in logit.max(1)[1]]

    def reset_parameters(self):
        self.model.load_state_dict(self._model_backup)
        if self.model.classifier.__class__.__name__ in \
                {"RobertaClassificationHead", "ElectraClassificationHead"}:
            self.model.classifier.dense.reset_parameters()
            self.model.classifier.out_proj.reset_parameters()
        elif hasattr(self.model.classifier, "reset_parameters"):
            self.model.classifier.reset_parameters()
        else:
            raise RuntimeError(
                f"parameter reset function not supported by "
                f"{self.model.classifier.__class__}")

    def trainable_parameters(self):
        return self.model.parameters()

    def reset(self):
        self.reset_parameters()

    def predict(self, data: Sequence[str],
                batch_size=32, **kwargs) -> Sequence[str]:
        with torch.no_grad():
            ret = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                preds = self._predict(batch)
                ret.extend(preds)
        return ret

    def fit(self, train_data: Dataset, valid_data: Dataset, **kwargs):
        optimizer = self.optimizer_cls(self.trainable_parameters())
        best_acc, best_epoch = None, None
        best_params = None

        valid_accs = []

        def _evaluate(data, event=None):
            with torch.no_grad():
                self.model.eval()

                if event is None:
                    logging.info("validating...")
                else:
                    logging.info(f"validating after {event}...")

                num_correct = 0
                dataloader = create_dataloader(
                    data,
                    batch_size=self.batch_size,
                    shuffle=False
                )
                for batch in dataloader:
                    preds = self.predict([ex.text for ex in batch])
                    num_correct += sum(
                        pred == ex.label for pred, ex in zip(preds, batch))
                acc = num_correct / len(data)
                valid_accs.append(acc)
            return acc

        for eidx in itertools.count(1):
            self.train()
            total_loss, total_size = 0.0, 0
            train_idx = list(range(len(train_data)))
            random.shuffle(train_idx)

            def _loss_update(b):
                nonlocal total_loss, total_size
                loss = self.compute_loss(b).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b)
                total_size += len(b)

            for idx in range(0, len(train_idx), self.batch_size):
                batch = train_data[train_idx[idx:idx + self.batch_size]]
                _loss_update(batch)

            avg_loss = total_loss / total_size
            acc = _evaluate(valid_data, f"{eidx}-th epoch")
            valid_accs.append(acc)

            logging.info(f"epoch: {eidx}, loss: {avg_loss}, valid_acc: {acc}")

            if best_acc is None or acc > best_acc:
                logging.info(f"best acc found @ epoch {eidx} (acc: {acc})")
                best_acc = acc
                best_epoch = eidx
                best_params = {k: v.detach().cpu().clone() for k, v in
                               self.state_dict().items()}

            if best_epoch is not None and eidx > best_epoch + self.patience:
                logging.info(f"early stop patience triggered @ epoch {eidx}")
                break

        if best_params is not None:
            logging.info(f"loading best parameters @ "
                         f"epoch {best_epoch} (acc: {best_acc})")
            self.load_state_dict(best_params)

        return {"best-epoch": best_epoch, "best-valid-acc": best_acc}
