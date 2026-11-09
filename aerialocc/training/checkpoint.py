"""Avoid external initialization downloads when restoring complete weights."""


def disable_external_initialization(model_config):
    """Modify a config before strict checkpoint restore, without changing layers."""
    if isinstance(model_config, dict):
        if 'init_cfg' in model_config:
            model_config['init_cfg'] = None
        if 'pretrained' in model_config:
            model_config['pretrained'] = None
        for value in model_config.values():
            disable_external_initialization(value)
    elif isinstance(model_config, (list, tuple)):
        for value in model_config:
            disable_external_initialization(value)
