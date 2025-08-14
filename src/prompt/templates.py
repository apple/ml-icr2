#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from typing import Any


class PromptTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} names.
    """

    def __init__(self, variables=None, template=None):
        self.variables = variables
        self.template = template

    def format(self, input_variables):
        """
        Returns the prompt using the `input_variables` in the form of {"query": "text", ...} to a string
        """
        return self.template.format(**input_variables)

    def get_template(self):
        return self.template


class PassageTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} for passage
    """

    def __init__(
        self, variables=["title_text", "passage_text"], template="- Title: {title_text}\n{passage_text}\n\n"
    ):
        self.variables = variables
        self.template = template

    def serialize_passages(self, passages):
        """
        Serializes the `passages` in the form of [{"context": "text"}, ...] to a string
        """
        return "".join(
            [self.template.format(**passage) for passage in passages]
        ).strip()

class CitePassageTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} for passage
    """

    def __init__(
        self, variables=["passage_id", "title_text", "passage_text"], template="- Passage ID: {passage_id}\n- Title: {title_text}\n{passage_text}\n\n"
    ):
        self.variables = variables
        self.template = template

    def serialize_passages(self, passages):
        """
        Serializes the `passages` in the form of [{"context": "text"}, ...] to a string
        """
        return "".join(
            [self.template.format(passage_id=str(i), title_text=passage['title_text'], passage_text=passage['passage_text']) for i, passage in enumerate(passages)]
        ).strip()
    

CONTEXT_CHUNK_TOKEN_START = "[CONTEXT]"
CONTEXT_CHUNK_TOKEN_END = "[/CONTEXT]"

class ICR2PassageTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} for passage
    """

    def __init__(
        self, variables=["passage_id", "title_text", "passage_text"], template=CONTEXT_CHUNK_TOKEN_START+"- Title: {title_text}\n{passage_text}\n\n"+CONTEXT_CHUNK_TOKEN_END
    ):
        self.variables = variables
        self.template = template

    def serialize_passages(self, passages):
        """
        Serializes the `passages` in the form of [{"context": "text"}, ...] to a string
        """
        return "".join(
            [self.template.format(title_text=passage['title_text'], passage_text=passage['passage_text']) for i, passage in enumerate(passages)]
        ).strip()

class HistoryTemplate:
    """
    Args:
        `templates` (dict{str: str}): The templates dictionary of 'speaker': 'speaker template'.
    """

    def __init__(self, templates={"Human": "User: {}\n", "Assistant": "Agent: {}\n"}):
        self.templates = templates

    def format_utterance(self, statement, speaker):
        assert speaker in self.templates, "{} is not a valid speaker.".format(speaker)
        return self.templates[speaker].format(statement)

    def serialize_history(self, history, max_history=10):
        """
        Serializes the `history` in the form of [{"speaker": "agent", "utterance": "text"}, ...] to a string
        """
        # remove from middle
        while len(history) > max_history:
            mid_point = len(history) // 2

            if mid_point % 2 == 0:
                history = history[: mid_point - 2] + history[mid_point:]
            else:
                history = history[: mid_point - 1] + history[mid_point + 1 :]

        return "".join(
            [
                self.format_utterance(context["utterance"], context["speaker"])
                for context in history
            ]
        ).strip()


class LLMEvalTemplate:
    """
    Args:
        `templates` (dict{str: str}): The templates dictionary of 'speaker': 'speaker template'.
    """

    def __init__(self, templates={"Human": "User: {}\n", "Assistant": "Agent: {}\n"}):
        self.templates = templates


class RAGQAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "Please answer the following question given the following passages:\n{retrieved_passages}\nQuestion: {query}\nAnswer: "
    
        self.passage_template = PassageTemplate()

    def set_document_id(self):
        self.passage_template = CitePassageTemplate()
    
    def set_context_special_token(self):
        self.passage_template = ICR2PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)
        prompt = self.format(
            {"query": sample['query_text'], "retrieved_passages": serialized_passages}
        )
        return prompt
    
class OracleRAGQAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "Please answer the following question given the following passages:\n{retrieved_passages}\nQuestion: {query}\nAnswer: "
        self.passage_template = PassageTemplate()
        self.corpus = None

    def build_oracle_corpus(self, corpus_path):
        import json
        from collections import defaultdict
        
        # corpus format: {pid: {title_text:, passage_text:}}
        self.corpus = defaultdict(dict)
        with open(corpus_path, "r") as f:
            for str_d in f.readlines():
                d = json.loads(str_d.strip())
                self.corpus[d['pid']] = d
        
    def extract_oracle_for_icr2(self, corpus):
        oracle_corpus = []
        for d in corpus:
            if d['model'] == "gold":
                oracle_corpus.append(d)
        self.corpus = oracle_corpus


    def __call__(self, sample):
        
        # Extracting oracle passages
        if "metadata" in sample.keys():
            oracle_psg_ids = sample['metadata']['qrels']
            passages = [self.corpus[idx] for idx, _ in oracle_psg_ids]
        else:
            passages = self.corpus

        # Padding oracle passages to prompt as the context
        if len(passages) > 0:
            serialized_passages = self.passage_template.serialize_passages(passages)
        else:
            serialized_passages = "" # handle the case corpus has no annotation
        prompt = self.format(
            {"query": sample['query_text'], "retrieved_passages": serialized_passages}
        )
        return prompt

class CloseBookQAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query"]
        self.template = "Please answer the following question:\nQuestion: {query}\nAnswer: "

        self.passage_template = PassageTemplate()

    def __call__(self, sample):
        prompt = self.format(
            {"query": sample['query_text']}
        )
        return prompt

class CloseBookFactCheckPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query"]
        self.template = "Please verify the given claim and predict your judgement on its factuality as TRUE or FALSE:\nClaim: {query}\nJudgement: "

        self.passage_template = PassageTemplate()

    def __call__(self, sample):
        prompt = self.format(
            {"query": sample['query_text']}
        )
        return prompt

class OracleRAGFactCheckPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "According to the following passages, please verify the given claim and predict your judgement on its factuality as TRUE or FALSE:\n{retrieved_passages}\nClaim: {query}\nJudgement: "
        self.passage_template = PassageTemplate()
        self.corpus = None

    def build_oracle_corpus(self, corpus_path):
        import json
        from collections import defaultdict
        
        # corpus format: {pid: {title_text:, passage_text:}}
        self.corpus = defaultdict(dict)
        with open(corpus_path, "r") as f:
            for str_d in f.readlines():
                d = json.loads(str_d.strip())
                self.corpus[d['pid']] = d
        
    def extract_oracle_for_icr2(self, corpus):
        oracle_corpus = []
        for d in corpus:
            if d['model'] == "gold":
                oracle_corpus.append(d)
        self.corpus = oracle_corpus


    def __call__(self, sample):
        
        # Extracting oracle passages
        if "metadata" in sample.keys():
            oracle_psg_ids = sample['metadata']['qrels']
            passages = [self.corpus[idx] for idx, _ in oracle_psg_ids]
        else:
            passages = self.corpus

        # Padding oracle passages to prompt as the context
        if len(passages) > 0:
            serialized_passages = self.passage_template.serialize_passages(passages)
        else:
            serialized_passages = "" # handle the case corpus has no annotation
        prompt = self.format(
            {"query": sample['query_text'], "retrieved_passages": serialized_passages}
        )
        return prompt

class RAGFactCheckPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "According to the following passages, please verify the given claim and predict your judgement on its factuality as TRUE or FALSE:\n{retrieved_passages}\nClaim: {query}\nJudgement: "
        self.passage_template = PassageTemplate()
    
    def set_document_id(self):
        self.passage_template = CitePassageTemplate()
    
    def set_context_special_token(self):
        self.passage_template = ICR2PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)
        prompt = self.format(
            {"query": sample['query_text'], "retrieved_passages": serialized_passages}
        )
        return prompt


class CloseBookDialoguePromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query"]
        self.template = "Please provide a single response to complete the following conversation by role-playing as either Person A or Person B. Your response should be as knowledgeable and coherent with the conversation history as possible.\nConversation:\n{query}"

        self.passage_template = PassageTemplate()

    def __call__(self, sample):

        query = sample['query_text'].split('\n')

        dialogue_history = ""
        role_A, role_B = "Person A: ", "Person B: "
        for i, q in enumerate(query):
            if i % 2 == 0:
                dialogue_history += role_A + q + '\n'
            elif i % 2 == 1:
                dialogue_history += role_B + q + '\n'
        
        if i % 2 == 0:
            dialogue_history += role_B 
        elif i % 2 == 1:
            dialogue_history += role_A 

        prompt = self.format(
            {"query": dialogue_history}
        )
        return prompt

class OracleRAGDialoguePromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "According to the given passages, please provide a single response to complete the following conversation by role-playing as either Person A or Person B. Your response should be as knowledgeable and coherent with the conversation history as possible.\nPassages:\n{retrieved_passages}\nConversation:\n{query}"
        self.passage_template = PassageTemplate()
        self.corpus = None

    def build_oracle_corpus(self, corpus_path):
        import json
        from collections import defaultdict
        
        # corpus format: {pid: {title_text:, passage_text:}}
        self.corpus = defaultdict(dict)
        with open(corpus_path, "r") as f:
            for str_d in f.readlines():
                d = json.loads(str_d.strip())
                self.corpus[d['pid']] = d
        
    def extract_oracle_for_icr2(self, corpus):
        oracle_corpus = []
        for d in corpus:
            if d['model'] == "gold":
                oracle_corpus.append(d)
        self.corpus = oracle_corpus


    def __call__(self, sample):
        
        # Extracting oracle passages
        if "metadata" in sample.keys():
            oracle_psg_ids = sample['metadata']['qrels']
            passages = [self.corpus[idx] for idx, _ in oracle_psg_ids]
        else:
            passages = self.corpus

        # Padding oracle passages to prompt as the context
        if len(passages) > 0:
            serialized_passages = self.passage_template.serialize_passages(passages)
        else:
            serialized_passages = "" # handle the case corpus has no annotation

        query = sample['query_text'].split('\n')

        dialogue_history = ""
        role_A, role_B = "Person A: ", "Person B: "
        for i, q in enumerate(query):
            if i % 2 == 0:
                dialogue_history += role_A + q + '\n'
            elif i % 2 == 1:
                dialogue_history += role_B + q + '\n'
        
        if i % 2 == 0:
            dialogue_history += role_B 
        elif i % 2 == 1:
            dialogue_history += role_A 

        prompt = self.format(
            {"query": dialogue_history, "retrieved_passages": serialized_passages}
        )
        return prompt
    

class RAGDialoguePromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "According to the given passages, please provide a single response to complete the following conversation by role-playing as either Person A or Person B. Your response should be as knowledgeable and coherent with the conversation history as possible.\nPassages:\n{retrieved_passages}\nConversation:\n{query}"
        self.passage_template = PassageTemplate()

    def set_document_id(self):
        self.passage_template = CitePassageTemplate()

    def set_context_special_token(self):
        self.passage_template = ICR2PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)

        query = sample['query_text'].split('\n')

        dialogue_history = ""
        role_A, role_B = "Person A: ", "Person B: "
        for i, q in enumerate(query):
            if i % 2 == 0:
                dialogue_history += role_A + q + '\n'
            elif i % 2 == 1:
                dialogue_history += role_B + q + '\n'
        
        if i % 2 == 0:
            dialogue_history += role_B 
        elif i % 2 == 1:
            dialogue_history += role_A 

        prompt = self.format(
            {"query": dialogue_history, "retrieved_passages": serialized_passages}
        )
        return prompt

class LlamaChatQAPromptTemplate(RAGQAPromptTemplate):
    def __init__(self):
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        self.variables = ["query", "retrieved_passages"]
        self.template = (
            self.B_INST
            + " "
            + self.B_SYS
            + "Please answer the following question given the following passages:"
            + self.E_SYS
            + "{retrieved_passages}\nQuestion: {query}\n"
            + self.E_INST
            + "\nAnswer: "
        )

        # Llama behaves wierdly at \n\n, so we modeify the passage template to not have \n\n
        self.passage_template = PassageTemplate(template="- Title: {title}\n{text}\n")


class LlamaSystemPromptTemplate(PromptTemplate):
    
    # Convert prompt to Mistral format with its system prompt
    # description: "Mistral 2 chat one shot prompt",
    # Mistral: "prompt": '''[INST] {instruction} [/INST]'''
    
    def __init__(self):
        self.B_INST = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 23 July 2024\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
'''
        self.E_INST = '''<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>'''

    def __call__(self, prompt) -> Any:
        return self.B_INST + ' ' + prompt + ' ' + self.E_INST

class MistralSystemPromptTemplate(PromptTemplate):
    
    # Convert prompt to Mistral format with its system prompt
    # description: "Mistral 2 chat one shot prompt",
    # Mistral: "prompt": '''[INST] {instruction} [/INST]'''
    
    def __init__(self):
        self.B_INST, self.E_INST = "[INST]", "[/INST]"

    def __call__(self, prompt) -> Any:
        return self.B_INST + ' ' + prompt + ' ' + self.E_INST

class QwenSystemPromptTemplate(PromptTemplate):
    
    # Convert prompt to Mistral format with its system prompt
    # description: "Mistral 2 chat one shot prompt",
    # Mistral: "prompt": '''[INST] {instruction} [/INST]'''
    
    def __init__(self):
        self.template = '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruct}<|im_end|>\n<|im_start|>assistant\n'''

    def __call__(self, prompt) -> Any:
        return self.template.format(instruct=prompt)

class PhiSystemPromptTemplate(PromptTemplate):
    
    # Convert prompt to Mistral format with its system prompt
    # description: "Phi 3 instruct one shot prompt",
    # Mistral: "prompt": '''<|user|>\n{instruct}<|end|>\n<|assistant|>'''
    
    def __init__(self):
        self.template = '''<|user|>\n{instruct}<|end|>\n<|assistant|>'''

    def __call__(self, prompt) -> Any:
        return self.template.format(instruct=prompt)


class QAUnaswerablePromptTemplate(RAGQAPromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = 'Please answer the following question given the following passage. If the answer is not in the passage or cannot be inferred from the passage, respond as "I don\'t know".\n{retrieved_passages}\nQuestion: {query}\nAnswer: '

        self.passage_template = PassageTemplate()


class LlamaChatQAUnaswerablePromptTemplate(LlamaChatQAPromptTemplate):
    def  __init__(self):
        super().__init__()
        self.template = (
            self.B_INST
            + " "
            + self.B_SYS
            + 'Please answer the following question given the following passages. If the answer is not in the passages or cannot be inferred from the passages, respond as "I don\'t know".'
            + self.E_SYS
            + "{retrieved_passages}\nQuestion: {query}\n"
            + self.E_INST
            + "\nAnswer: "
        )


class ConvQAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages", "history"]
        self.template = "Please answer the following question given the following passages and the conversation history:\n\n{retrieved_passages}\n\n{history}\nUser: {query}\nAgent: "

        self.history_template = HistoryTemplate()
        self.passage_template = PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)
        serialized_history = self.history_template.serialize_history(sample.context)
        prompt = self.format(
            {
                "query": sample.question,
                "retrieved_passages": serialized_passages,
                "history": serialized_history,
            }
        )
        return prompt


class LlamaChatConvQAPromptTemplate(ConvQAPromptTemplate):
    def __init__(self):
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        self.variables = ["query", "retrieved_passages", "history"]
        self.template = (
            self.B_INST
            + " "
            + self.B_SYS
            + "Please answer the following question given the following passages and the conversation history:"
            + self.E_SYS
            + "{retrieved_passages}\n{history}\nuser: {query}\n"
            + self.E_INST
            + "\nassistant: "
        )

        self.history_template = HistoryTemplate(templates={"Human": "user: {}\n", "Assistant": "assistant: {}\n"})
        self.passage_template = PassageTemplate(template="- Title: {title}\n{text}\n")



class ConvQAUnaswerablePromptTemplate(ConvQAPromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages", "history"]
        self.template = 'Please answer the following question given the following passage and the conversation history. If the answer is not in the passage or cannot be infered from the passage, respond as "I don\'t know".\n\n{retrieved_passages}\n\n{history}\nUser: {query}\nAgent: '

        self.history_template = HistoryTemplate()
        self.passage_template = PassageTemplate()


class LlamaChatConvQAUnaswerablePromptTemplate(LlamaChatConvQAPromptTemplate):
    def __init__(self):
        super().__init__()
        self.template = (
            self.B_INST
            + " "
            + self.B_SYS
            + 'Please answer the following question given the following passages and the conversation history. If the answer is not in the passages or cannot be infered from the passages, respond as "I don\'t know".'
            + self.E_SYS
            + "{retrieved_passages}\n{history}\nuser: {query}\n"
            + self.E_INST
            + "\nassistant: "
        )
