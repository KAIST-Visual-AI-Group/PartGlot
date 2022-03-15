from typing import Dict
from collections import defaultdict
from partglot.utils.simple_utils import sort_and_count_len_of_dict_values
import numpy as np
from pandas import DataFrame, Series


part_names = ["back", "seat", "leg", "arm"]

part_semantic_groups = {
    "back": ["back"],
    "seat": ["seat"],
    "leg": ["leg", "wheel", "base"],
    "arm": ["arm"],
}


def convert_labels_to_one_hot(labels, n_classes=3):
    # Convert labels to indicators.
    n_utter = len(labels)
    targets = np.array(labels, dtype=np.int64)
    target_oh = np.zeros(shape=(n_utter, n_classes))
    for i in range(n_utter):
        target_oh[i, targets[i]] = 1
    assert np.all(np.sum(target_oh, axis=1) == 1)
    return target_oh


def get_part_indicator(text: Series, word2int: Dict):
    part_indicator = np.zeros((len(text), len(part_names)), dtype=np.float32)

    part_idx_set = dict()
    for k, v in part_semantic_groups.items():
        part_idx_set[k] = set(list(map(lambda x:word2int[x], v)))
        
    for i in range(len(text)):
        token_set = set(text[i])
        for j, part_name in enumerate(part_names):
            current_set = part_idx_set[part_name]
            other_set = set.union(*part_idx_set.values()) - current_set
            if len(set.intersection(token_set, current_set)) and not len(
                set.intersection(token_set, other_set)
            ):
                part_indicator[i, j] = 1

    part_mask = np.sum(part_indicator, 1) > 0

    return part_indicator, part_mask


def get_mask_of_game_data(
    game_data: DataFrame,
    word2int: Dict,
    only_correct: bool,
    only_easy_context: bool,
    max_seq_len: int,
    only_one_part_name: bool,
):
    """
    only_correct (if True): mask will be 1 in location iff human listener predicted correctly.
    only_easy (if True): uses only easy context examples (more dissimilar triplet chairs)
    max_seq_len: drops examples with len(utterance) > max_seq_len
    only_one_part_name (if True): uses only utterances describing only one part in the give set.
    """
    mask = np.array(game_data.correct)
    if not only_correct:
        mask = np.ones_like(mask, dtype=np.bool)

    if only_easy_context:
        context_mask = np.array(game_data.context_condition == "easy", dtype=np.bool)
        mask = np.logical_and(mask, context_mask)

    short_mask = np.array(
        game_data.text.apply(lambda x: len(x)) <= max_seq_len, dtype=np.bool
    )
    mask = np.logical_and(mask, short_mask)

    part_indicator, part_mask = get_part_indicator(game_data.text, word2int)
    if only_one_part_name:
        mask = np.logical_and(mask, part_mask)

    return mask, part_indicator


def shuffle_game_geometries(geo_ids, labels, parts=None, random_seed=None):
    """e.g. if [a, b, c] with label 1 makes it [b, a, c] with label 0."""
    if random_seed is not None:
        np.random.seed(random_seed)
    shuffle = np.random.shuffle
    for i in range(len(geo_ids)):
        idx = [0, 1, 2]
        shuffle(idx)
        geo_ids[i] = geo_ids[i][idx]
        labels[i] = labels[i][idx]
        if parts is not None:
            parts[i] = parts[i][idx]

    if parts is not None:
        return geo_ids, labels, parts
    else:
        return geo_ids, labels


def pad_text_symbols_with_zeros(
    text, max_seq_len, dtype=np.int64, force_zero_end=False
):
    """
    force_zero_end (bool) if True every sequence will end with zero, alternatively,
        sequences with equal or more elements than max_seq_len will end with the element at that max_seq_len position.
    """
    text_padded = []
    seq_len = []

    if force_zero_end:
        last_slot = 1
    else:
        last_slot = 0

    for sentence in text:
        pad_many = max_seq_len - len(sentence) + last_slot
        if pad_many > 0:
            text_padded.append(
                np.pad(sentence, (0, pad_many), "constant", constant_values=0)
            )
            seq_len.append(len(sentence))
        else:
            keep_same = min(max_seq_len, len(sentence))
            kept_text = sentence[:keep_same]
            if force_zero_end:
                kept_text.append(0)
            text_padded.append(kept_text)
            seq_len.append(keep_same)

    text_padded = np.array(text_padded, dtype=dtype)
    seq_len = np.array(seq_len, dtype)
    return text_padded, seq_len


def group_geometries(geo_ids, target_indicators=None):
    """If the geometries associated with an utterance are the same, group them together.
    Input:
        geo_ids (N x 3): N triplets of integers.
        target_indicators: (N x 3) indicator of which was the target geometry for each of the geo_ids.
    Returns:
         A dictionary where each key maps to the rows of the ``geo_ids`` that are comprised by the same set of integers, and same ``target_indicators``, if the latter is not None.
    """
    if target_indicators is not None and not np.all(
        np.unique(target_indicators) == [0, 1]
    ):
        raise ValueError("provide one-hot indicators.")

    n_triplets = len(geo_ids)
    if target_indicators is not None:
        if n_triplets != len(target_indicators):
            raise ValueError()

    groups = defaultdict(list)
    for i in range(n_triplets):
        g = geo_ids[i].astype(np.int)
        if target_indicators is not None:
            t = g[np.where(target_indicators[i])]
            key = tuple(np.hstack([sorted(g), t]))
        else:
            key = tuple(sorted(g))
        groups[key].append(i)
    return groups


def group_target_geometries(geo_ids, indicators):
    """Returns a dictionary mapping each geo_id at the rows of geo_ids of which it was
    used as a target. I.e., the values of each key in the result are an equivalence class.
    """

    if not np.all(np.unique(indicators) == [0, 1]):
        raise ValueError("provide one-hot indicators.")

    groups = defaultdict(list)  # Hold for each target_geometry the rows which is used.
    n_triplets = len(geo_ids)
    for i in range(n_triplets):
        target_geo_i = geo_ids[i][np.where(indicators[i])][0]
        groups[target_geo_i].append(i)
    return groups


def split_indices_with_unseen_target_geo_in_test(
    loads, geo_ids, labels, bias_train=False, seed=None, debug=True
):
    """Args:
        loads: (list) train-test-val split percentages strictly positive and must sum to 1.0.
        geo_ids: (n x 3) triplets of geo_ids for n game-interactions.
        labels:  (n x 3) indicators of target geometries.
        bias_train: (boolean) if True, then the training examples will consists of targets with
    more super-vised utterances than the test/val.
    """
    train_per, test_per, val_per = loads
    if np.sum(loads) != 1.0:
        raise ValueError("train-test-val split must sum to 1.0")

    target_classes = group_target_geometries(geo_ids, labels)

    # Count number of utterances associated with each target geometry.
    # (To push triplets with "a lot" of utterances into trainining.)
    sorted_target_classes, lengths = sort_and_count_len_of_dict_values(target_classes)
    sorted_target_classes = sorted_target_classes.astype(np.int32)

    if bias_train:
        p = lengths / lengths.sum()
    else:
        p = None

    if seed is not None:
        np.random.seed(seed)

    n_targets = len(target_classes)
    train_size = int(np.ceil(train_per * n_targets))
    rest_size = n_targets - train_size
    test_size = int(n_targets * test_per)
    val_size = rest_size - test_size
    assert val_size + train_size + test_size == n_targets
    train_ids = np.random.choice(sorted_target_classes, train_size, replace=False, p=p)
    rest_ids = np.setdiff1d(sorted_target_classes, train_ids)
    test_ids = np.random.choice(rest_ids, test_size, replace=False, p=None)
    val_ids = np.setdiff1d(rest_ids, test_ids)

    # Back from target geo-ids to rows.
    res = {}
    for s, ids in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        rows = []
        for i in ids:
            rows.extend(target_classes[i])
        res[s] = np.array(rows)

    if debug:
        set_geos = defaultdict(list)

        for s in ["train", "test", "val"]:
            for i, l in zip(geo_ids[res[s]], labels[res[s]]):
                set_geos[s].append(i[np.where(l)[0]][0])
            set_geos[s] = set(set_geos[s])

        assert (
            set_geos["train"].isdisjoint(set_geos["test"])
            and set_geos["val"].isdisjoint(set_geos["test"])
            and set_geos["train"].isdisjoint(set_geos["val"])
        )

        # print ('unique geometries in train/test/val',
        # len(set_geos['train']), len(set_geos['test']), len(set_geos['val']))

    return res


def rotation(pc):
    rot = np.array([[0, 0, -1], [0, 1, 0], [-1, 0, 0.0]])
    rot_pc = pc @ rot
    return rot_pc


def get_pc_norm_info_each_sample(geos):
    k, l, _ = geos.shape

    normed_geos = geos.copy()
    centroids = np.mean(normed_geos, axis=1, keepdims=True)  # [3, 1, 3]

    normed_geos = normed_geos - centroids

    m = np.max(np.sqrt(np.sum(normed_geos ** 2, axis=-1)))

    return centroids, m


