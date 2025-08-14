#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from evaluator.metrics import (
    Bleu,
    BERTScore,
    EM,
    F1,
    LLMEval,
    LLMEvalConv,
    Meteor,
    Recall,
    RecallEM,
    Precision,
    Rouge,
    FaithDialCriticInverse,
    PolarizedRecallEM,
    BidirectRecallEM,
    RetrievalRecallCCI,
    RetrievalRecallRTA,
    RetrievalRecallTopK,
    ProbeAttentionTopK
)

# from evaluator.faithfulness_metrics import FaithDialCritic, FaithDialCriticV2, QSquared, KBERTScore, KF1, KF1PlusPlus, KPrecision, KPrecisionPlusPlus, KRecall, KRecallPlusPlus, KLLMEval, KLLMEvalConv


def load_metric(name, file_name=None, args=None):
    metric_mapping = {
        "meteor": Meteor,
        "rouge": Rouge,
        "f1": F1,
        "bleu": Bleu,
        "em": EM,
        "recall": Recall,
        "recallem": RecallEM,
        "bidirect_recallem": BidirectRecallEM,
        "polarizedem": PolarizedRecallEM,
        "precision": Precision,
        "bertscore": BERTScore,
        "llm_eval": LLMEval,
        "llm_eval_conv": LLMEvalConv,
        "faithcritic_inverse": FaithDialCriticInverse,
        "retrieval_recall_cci": RetrievalRecallCCI,
        "retrieval_recall_rta": RetrievalRecallRTA,
        "retrieval_recall_TopK": RetrievalRecallTopK,
        "retrieval_probe_attention": ProbeAttentionTopK,
    }

    if name not in metric_mapping:
        raise ValueError(f"{name} is not a valid metric.")

    return metric_mapping[name](name, file_name=file_name, args=args)

