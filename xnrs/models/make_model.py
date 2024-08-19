from .components import scoring
from .full_models import (
    StandardRec,
    BaseRec, 
    MeanRec,
    ParamFreeRec,
    CAUM,
    LSTUR,
    NPA,
    NRMS,
    NAML, SmallNAML
)


def make_model(cfg):

    emb_dim = cfg.total_emb_dim
    bias = cfg.bias

    # make scoring
    if cfg.scoring=='dot':
        scoring_fn = scoring.DotScoring()
    elif cfg.scoring=='bilin':
        scoring_fn = scoring.BilinScoring(emb_dim, bias=bias)
    elif cfg.scoring=='nonlin':
        scoring_fn = scoring.NonLinScoring(emb_dim, bias=bias)
    elif cfg.scoring=='fc':
        scoring_fn = scoring.FCScoring(emb_dim, hidden_dim = emb_dim // 2, bias=bias)
    elif cfg.scoring=='CAUMScoring':
        scoring_fn = scoring.CAUMScoring()
    else:
        raise ValueError(f'invalid value for cfg.scoring: {cfg.scoring}')

    # make base model
    if cfg.model=='standard':
        model = StandardRec(cfg, scoring_fn)
    elif cfg.model=='base':
        model = BaseRec(cfg, scoring_fn)
    elif cfg.model=='mean':
        model = MeanRec(cfg, scoring_fn)
    elif cfg.model=='NRMS':
        model = NRMS(cfg, scoring_fn)
    elif cfg.model=='NAML':
        model = NAML(cfg, scoring_fn)
    elif cfg.model=='smallNAML':
        model = SmallNAML(cfg, scoring_fn)
    elif cfg.model=='NPA':
        model = NPA(cfg, scoring_fn)
    elif cfg.model=='LSTUR':
        model = LSTUR(cfg, scoring_fn)
    elif cfg.model=='CAUM':
        model = CAUM(cfg, scoring_fn)
    else:
        raise ValueError(f'invalid value for cfg.model: {cfg.model}')

    return model