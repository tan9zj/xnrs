import torch
from tqdm import tqdm
from typing import Optional
from pandas import DataFrame


def collect_features(cfg):
    features = []
    if cfg.text_features:
        features += cfg.text_features
    if cfg.catg_features:
        assert cfg.base_model in ['category', 'NAML', 'smallNAML', 'LSTUR', 'CAUM'],\
            f'attempting to load categorical features for {cfg.base_model} model'
        features += cfg.catg_features
    if cfg.add_features:
        features += cfg.add_features
    # if cfg.weight_loss == True:
    #     features.append('clicks')
    assert features, 'no features are defined'
    return features


def tokenize(text, tokenizer, seq_len: int = 50, build_reference: bool = False):
    tokens = tokenizer(
            text, 
            max_length=seq_len,
            padding='max_length',
            truncation=True, 
            return_tensors='pt'
        )
    if build_reference:  # reference of padding tokens with cls and eos 
        pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        eos_idx = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        # sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        ref = torch.full_like(tokens.input_ids, pad_idx)
        ref[0][0] = cls_idx
        ref[0][(tokens.input_ids != pad_idx).sum().item() - 1] = eos_idx
        # ref[0][(tokens.input_ids != pad_idx).sum().item() - 1] = sep_idx
        return tokens, ref
    else:
        return tokens, None


def compute_embedding(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reference: Optional[torch.tensor] = None,
    device: torch.device = torch.device('cuda:0')
):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if reference is not None: 
        reference = reference.to(device)
        input_ids = torch.cat([input_ids, reference], dim=0)
        attention_mask = attention_mask.repeat(2, 1)
    emb = model(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device)
        ).last_hidden_state
    if reference is not None:
        emb = emb[:-1] - emb[-1]
        attention_mask = attention_mask[:-1]
        input_ids = input_ids[:-1]
    return emb, attention_mask, input_ids


def precompute_embeddings(
    texts: list, 
    tokenizer, model, 
    seq_len: int,
    reduction: str = 'none',
    device: torch.device = torch.device('cuda:0'),
    return_tokens: bool = False,
    relative_to_reference: bool = True
):
    model.eval()
    model.to(device)
    embeddings = []
    all_tokens = []
    with torch.no_grad():
        for text in tqdm(texts):
            tokens, ref = tokenize(
                text, 
                tokenizer=tokenizer, 
                seq_len=seq_len,
                build_reference=relative_to_reference
                )
            emb, att, tkn = compute_embedding(
                model=model,
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                reference=ref,
                device=device
            )
            if reduction == 'mean':
                reduced = torch.mean(emb, dim=0)
                att_reduced = torch.clamp(torch.sum(att, dim=0), 0, 1)
            elif reduction == 'cls':
                reduced = emb[0]
                att_reduced = att[0]
            elif reduction == 'none':
                reduced = emb
                att_reduced = att
            else:
                raise ValueError('reduction can be: none, mean or cls')
            embeddings.append((
                reduced.detach().cpu().numpy(), 
                att_reduced.detach().cpu().numpy()
            ))
            all_tokens.append(tkn)
    if return_tokens:
        return embeddings, all_tokens
    else:
        return embeddings
    

def index_category(data: DataFrame, column: str, category_idx: Optional[dict] = None,
    return_category_idx: bool = False, dtype='numpy'):
    assert column in data.columns
    assert not data[column].isna().any()
    categories = sorted(list(data[column].unique()))
    if category_idx is None:  # index all categories
        category_idx = {
            c: i + 1 for i, c in enumerate(categories)
        }
    else:  # assign index 0 to all unknown categories
        for c in categories:
            if c not in category_idx.keys():
                category_idx[c] = 0
    data[column + '_index'] = data[column].map(lambda c: category_idx[c])
    if return_category_idx:
        return data, category_idx
    else:
        return data