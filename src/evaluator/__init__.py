#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import json
import os
import re
import string

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name

    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()

    @classmethod
    def _normalize_text(cls, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = text.lower()
        text = text.replace(" 's", "'s").replace(" - ", "-").replace(" : ", ":")
        text = "".join(char for char in text if char not in set(string.punctuation))
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def _get_tokens(self, text):
        if not text:
            return []
        return self._normalize_text(text).split()

    
