import numpy as np
import torch


def init_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# Random arrays
def shuffle(*arrays, **kwargs):
    # indices? Whether return the shuffle_indices(True/False)
    require_indices = kwargs.get('indices', False)

    # Determine whether the lengths are equal. If equal, the length of the set is 1
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('Inputs to shuffles must be have the same length')

    # We set seed before, so we can shuffle the index in the arrays(csr_matrix)
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def mini(*arrays, **kwargs):
    batch_size = kwargs.get('batch_size', 2048)

    if len(arrays) == 1:
        for i in range(0, len(arrays[0]), batch_size):
            yield arrays[0][i: i + batch_size]
    else:
        for i in range(0, len(arrays[0]), batch_size):
            yield tuple(array[i: i + batch_size] for array in arrays)


def local_space_generator(awn, local_space_scale):
    return LocalSpace.generate(awn, local_space_scale)


def convert_sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    value = torch.FloatTensor(coo.data)
    sp_tensor = torch.sparse.FloatTensor(index, value, torch.Size(coo.shape))
    sp_tensor = sp_tensor.coalesce()
    return sp_tensor


class LocalSpace(object):
    def __init__(self, awn, local_space_scale):
        self.awn = awn
        self.local_space_scale = local_space_scale
        self.capacity = min(local_space_scale, len(awn))

        self.anchor_user_space = torch.unique(awn[:, 0])
        self.neighbor_item_space = torch.unique(awn[:, 1])

        self.anchor_space = self._unique_non_repeated(awn[:, 0])
        self.neighbor_space = self._unique_non_repeated(awn[:, 1])

        self.uniform_user_space = self.anchor_space
        self.uniform_item_space = self.neighbor_space

    @staticmethod
    def _unique_non_repeated(ids):
        unique_ids, _, counts = torch.unique(ids, return_inverse=True, return_counts=True)
        return unique_ids[counts == 1]

    @classmethod
    def generate(cls, awn, local_space_scale):
        for i in range(0, len(awn), local_space_scale):
            yield cls(awn[i: i + local_space_scale], local_space_scale)

    @classmethod
    def sample_from_dataset(cls, dataset, local_space_scale, device):
        awn = torch.Tensor(dataset.random_create_awn()).long()
        awn = shuffle(awn).to(device)

        num_local_spaces = len(awn) // local_space_scale + 1
        return cls.generate(awn, local_space_scale), num_local_spaces
