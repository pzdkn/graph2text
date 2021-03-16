from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer)
t5_models = ['t5-small', 't5-base']


def load_model_and_tokenizer(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)
    tokenizer.add_special_tokens({'additional_special_tokens': ['%SEP%', '%EOF%']})
    assert '%SEP%' in tokenizer.additional_special_tokens and '%EOF%' in tokenizer.additional_special_tokens
    model.resize_token_embeddings(len(tokenizer))
    # add to device ??
    return model, tokenizer
