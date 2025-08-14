#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from prompt import PassageTemplate, RAGQAPromptTemplate, LlamaChatQAPromptTemplate, QAUnaswerablePromptTemplate, LlamaChatQAUnaswerablePromptTemplate, ConvQAPromptTemplate, LlamaChatConvQAPromptTemplate, ConvQAUnaswerablePromptTemplate, LlamaChatConvQAUnaswerablePromptTemplate, MistralSystemPromptTemplate, QwenSystemPromptTemplate, PhiSystemPromptTemplate, CloseBookQAPromptTemplate, OracleRAGQAPromptTemplate, CloseBookFactCheckPromptTemplate, OracleRAGFactCheckPromptTemplate, RAGFactCheckPromptTemplate, RAGDialoguePromptTemplate, CloseBookDialoguePromptTemplate, OracleRAGDialoguePromptTemplate, LlamaSystemPromptTemplate

def load_template(template_name):
    """
    Loads template by name.

    Args:
        template_name (str): Name of template to load.

    Returns:
        PromptTemplate: Template object.
    """
    template_mapping = {
        "passage": PassageTemplate,
        "rag-qa": RAGQAPromptTemplate,
        "closebook-qa": CloseBookQAPromptTemplate,
        "oracle-qa": OracleRAGQAPromptTemplate,
        "rag-fact-check": RAGFactCheckPromptTemplate,
        "closebook-fact-check":CloseBookFactCheckPromptTemplate, 
        "oracle-fact-check": OracleRAGFactCheckPromptTemplate, 
        "rag-dialogue": RAGDialoguePromptTemplate,
        "closebook-dialogue":CloseBookDialoguePromptTemplate, 
        "oracle-dialogue": OracleRAGDialoguePromptTemplate, 
        "qa_unanswerable": QAUnaswerablePromptTemplate,
        "conv_qa": ConvQAPromptTemplate,
        "conv_qa_unanswerable": ConvQAUnaswerablePromptTemplate,
        "llama_chat_qa": LlamaChatQAPromptTemplate,
        "llama_chat_qa_unanswerable": LlamaChatQAUnaswerablePromptTemplate,
        "llama_chat_conv_qa": LlamaChatConvQAPromptTemplate,
        "llama_chat_conv_qa_unanswerable": LlamaChatConvQAUnaswerablePromptTemplate,
        "mistral_sys": MistralSystemPromptTemplate,
        "qwen_sys": QwenSystemPromptTemplate,
        "phi_sys": PhiSystemPromptTemplate,
        "llama_sys": LlamaSystemPromptTemplate,
    }

    if template_name not in template_mapping:
        raise ValueError(f"{template_name} is not a valid template.")

    return template_mapping[template_name]()
