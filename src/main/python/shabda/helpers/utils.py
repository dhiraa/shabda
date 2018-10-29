from shabda.hyperparams.hyperparams import HParams

def maybe_hparams_to_dict(hparams):
    """If :attr:`hparams` is an instance of :class:`~texar.HParams`,
    converts it to a `dict` and returns. If :attr:`hparams` is a `dict`,
    returns as is.
    """
    if hparams is None:
        return None
    if isinstance(hparams, dict):
        return hparams
    return hparams.todict()

def dict_fetch(src_dict, tgt_dict_or_keys):
    """Fetches a sub dict of :attr:`src_dict` with the keys in
    :attr:`tgt_dict_or_keys`.

    Args:
        src_dict: A dict or instance of :class:`~texar.HParams`.
            The source dict to fetch values from.
        tgt_dict_or_keys: A dict, instance of :class:`~texar.HParams`,
            or a list (or a dict_keys) of keys to be included in the output
            dict.

    Returns:
        A new dict that is a subdict of :attr:`src_dict`.
    """
    if src_dict is None:
        return src_dict

    if isinstance(tgt_dict_or_keys, HParams):
        tgt_dict_or_keys = tgt_dict_or_keys.todict()
    if isinstance(tgt_dict_or_keys, dict):
        tgt_dict_or_keys = tgt_dict_or_keys.keys()
    keys = list(tgt_dict_or_keys)

    if isinstance(src_dict, HParams):
        src_dict = src_dict.todict()

    return {k: src_dict[k] for k in keys if k in src_dict}