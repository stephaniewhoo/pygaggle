from pathlib import Path
from typing import List, Optional
import abc
import numpy as np

from pygaggle.data.relevance import RelevanceExample

__all__ = ['Writer', 'MsMarcoWriter', 'TrecWriter', 'HiddenTrecWriter']


class Writer:
    def __init__(self, path: Optional[Path] = None, overwrite: bool = True, tag: Optional[str] = None):
        self.to_output = str(path) not in [".", None]
        print(f'Writing run: {self.to_output}')
        if self.to_output:
            self.f = open(path, "w" if overwrite else "w+")
        self.tag = tag

    def write_line(self, text: str):
        if self.to_output:
            self.f.write(f"{text}\n")

    @abc.abstractmethod
    def write(self, scores: List[float], example: RelevanceExample):
        pass


class MsMarcoWriter(Writer):
    def write(self, scores: List[float], example: RelevanceExample):
        doc_scores = sorted(list(zip(example.documents, scores)),
                            key=lambda x: x[1], reverse=True)
        for ct, (doc, score) in enumerate(doc_scores):
            self.write_line(f"{example.query.id}\t{doc.metadata['docid']}\t{ct+1}")

class TrecWriter(Writer):
    def write(self, scores: List[float], example: RelevanceExample):
        doc_scores = sorted(list(zip(example.documents, scores)),
                            key=lambda x: x[1], reverse=True)
        for ct, (doc, score) in enumerate(doc_scores):
            self.write_line(f"{example.query.id}\tQ0\t{doc.metadata['docid']}\t{ct+1}\t{score}\t{self.tag}")

class HiddenTrecWriter(Writer):
    def __init__(self, path: Optional[Path] = None, overwrite: bool = True, tag: Optional[str] = None):
        self.to_output = str(path) not in [".", None]
        print(f'Writing run: {self.to_output}')
        if self.to_output:
            self.textf = open(str(path)+".trec", "w" if overwrite else "w+")
            self.binf = open(str(path)+".bin", "wb" if overwrite else "wb+") 
        self.tag = tag

    def write(self, scores: List[float], example: RelevanceExample):
        doc_scores = sorted(list(zip(example.documents, scores)),
                            key=lambda x: x[1], reverse=True)
        for ct, (doc, score) in enumerate(doc_scores):
            self.textf.write(f"{example.query.id}\tQ0\t{doc.metadata['docid']}\t{ct+1}\t{score}\t{self.tag}\n")
            hidden = doc.metadata["pooled_output"]
            assert hidden.dtype == np.float32
            self.binf.write(hidden.tobytes())
