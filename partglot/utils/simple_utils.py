"""
The MIT License (MIT)
Originally created sometime in late 2018.
Copyright (c) 2021 Panos Achlioptas (ai.stanford.edu/~optas) & Stanford Geometric Computing Lab
"""

import operator
import numpy as np
from six.moves import cPickle
from typing import Sequence
from omegaconf import DictConfig, OmegaConf
import rich.syntax
import rich.tree


def invert_dictionary(d):
    inv_map = {v: k for k, v in d.items()}
    return inv_map


def sort_dict_by_val(in_dict, reverse=False):
    return sorted(list(in_dict.items()), key=operator.itemgetter(1), reverse=reverse)


def sort_dict_by_key(in_dict, reverse=False):
    return sorted(list(in_dict.items()), key=operator.itemgetter(0), reverse=reverse)


def unique_rows(a, perm_free=False):
    if perm_free:
        return np.vstack(list({tuple(sorted(row)) for row in a}))
    else:
        return np.vstack(list({tuple(row) for row in a}))


def sort_and_count_len_of_dict_values(in_dictionary):
    n_elements = len(in_dictionary)
    lengths = np.zeros(n_elements)
    sorted_keys = np.zeros(n_elements)
    c = 0
    for key, val in sort_dict_by_key(in_dictionary):
        lengths[c] = len(val)
        sorted_keys[c] = key
        c += 1

    return sorted_keys, lengths


def iterate_in_chunks(l, n):
    """Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file."""
    out_file = open(file_name, "wb")
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: a generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """

    in_file = open(file_name, "rb")
    if python2_to_3:
        size = cPickle.load(in_file, encoding="latin1")
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding="latin1")
        else:
            yield cPickle.load(in_file)
    in_file.close()


def get_iou_per_instance_per_part(
    pred: np.ndarray, groundtruth: np.ndarray, num_part: int
):
    groundtruth = groundtruth.reshape(-1)
    pred = pred.reshape(-1)

    assert groundtruth.shape == pred.shape

    iou = np.zeros(num_part)

    for i in range(num_part):
        p = pred == i
        gt = groundtruth == i

        union = (gt | p).sum()
        inter = (gt & p).sum()
        if union == 0:
            iou[i] = 1
        else:
            iou[i] = inter / union

    return iou


def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)
