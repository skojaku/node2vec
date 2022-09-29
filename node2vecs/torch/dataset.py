import numpy as np
from collections import Counter
import numpy as np
from numba import njit
import numpy as np
from torch.utils.data import Dataset

#
# Triplet sampler
#
class NegativeSamplingDataset(Dataset):
    def __init__(
        self,
        seqs,
        window,
        epochs=1,
        buffer_size=1000,
        num_negative_samples=5,
        context_window_type="double",
        ns_exponent=1,
    ):
        self.window = window
        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.neg_sample_counter = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None
        self.seqs = seqs
        self.epochs = epochs
        self.context_window_type = context_window_type
        self.num_negative_samples = num_negative_samples

        # Count sequence elements
        counter = Counter()
        self.n_samples = 0
        for seq in seqs:
            counter.update(np.array(seq))
            n_pairs = count_center_context_pairs(window, len(seq), context_window_type)
            self.n_samples += n_pairs
        self.n_elements = int(max(counter.keys()) + 1)
        self.ele_null_prob = np.zeros(self.n_elements)
        for k, v in counter.items():
            self.ele_null_prob[k] = v
        self.ele_null_prob = np.power(self.ele_null_prob, ns_exponent)
        self.ele_null_prob /= np.sum(self.ele_null_prob)
        self.n_seqs = len(seqs)

        # Initialize
        self.iter = iter(seqs)
        self._generate_samples()

    def __len__(self):
        return self.n_samples * self.epochs

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        if self.neg_sample_counter == 0:
            center = self.centers[self.sample_id]
            cont = self.contexts[self.sample_id].astype(np.int64)
            y = 1
            self.neg_sample_counter = self.num_negative_samples
            return center, cont, 1
        else:
            center = self.centers[self.sample_id]
            cont = np.random.choice(self.n_elements, p=self.ele_null_prob)
            y = -1
            self.neg_sample_counter -= 1  # decrement the counter
            if self.neg_sample_counter == 0:
                self.sample_id += 1

        return center, cont, y

    def _generate_samples(self):
        self.centers = []
        self.contexts = []
        for _ in range(self.buffer_size):

            seq = next(self.iter, None)
            if seq is None:
                self.iter = iter(self.seqs)
                seq = next(self.iter, None)

            cent, cont = _get_center_context_pairs(
                np.array(seq),
                self.window,
                self.context_window_type,
            )
            self.centers.append(cent)
            self.contexts.append(cont)
        self.centers, self.contexts = (
            np.concatenate(self.centers),
            np.concatenate(self.contexts),
        )
        self.n_sampled = len(self.centers)
        self.sample_id = 0


class ModularityEmbeddingDataset(NegativeSamplingDataset):
    def __init__(self, **params):
        super(ModularityEmbeddingDataset, self).__init__(**params)
        self.center, self.cont, self.y = None, None, None

    def __getitem__(self, idx):
        pow = 1
        if self.center is None:
            center, cont, y = super(ModularityEmbeddingDataset, self).__getitem__(idx)
            if y == 1:
                self.center, self.cont, self.y = center, cont, y
                center = np.random.choice(self.n_elements)
                cont = np.random.choice(self.n_elements)
                y = -0.5
                pow = 2
        else:
            center, cont, y = self.center, self.cont, self.y
            self.center, self.cont, self.y = None, None, None
        return center, cont, y, pow


@njit(nogil=True)
def count_center_context_pairs(window, seq_len, context_window_type):
    # Count the number of center-context pairs.
    # Suppose that we sample, for each center word, a context word that proceeds the center k-words.
    # There are T-k words in the sequence, so that the number of pairs is given by summing this over k upto min(T-1, L), i.e.,
    # 2 \sum_{k=1}^{min(T-1, L)} (T-k)
    # where we cap the upper range to T-1 in case that the window covers the entire sentence, and
    # we double the sum because the word window extends over both proceeding and succeeding words.
    min_window = np.minimum(window, seq_len - 1)
    n = 2 * min_window * seq_len - min_window * (min_window + 1)
    if context_window_type == "double":
        return int(n)
    else:
        return int(n / 2)


@njit(nogil=True)
def _get_center_context_pairs(seq, window, context_window_type):
    """Get center-context pairs from sequence.
    :param seq: Sequence
    :type seq: numpy array
    :param window_length: Length of the context window
    :type window_length: int
    :return: Center, context pairs
    :rtype: tuple
    """
    n_seq = len(seq)
    n_pairs = count_center_context_pairs(window, n_seq, context_window_type)
    centers = -np.ones(n_pairs, dtype=np.int64)
    contexts = -np.ones(n_pairs, dtype=np.int64)
    idx = 0
    wstart, wend = 0, 2 * window + 1
    if context_window_type == "suc":
        wstart = window + 1
    if context_window_type == "prec":
        wend = window

    for i in range(n_seq):
        for j in range(wstart, wend):
            if (j != window) and (i - window + j >= 0) and (i - window + j < n_seq):
                centers[idx] = seq[i]
                contexts[idx] = seq[i - window + j]
                idx += 1
    return centers, contexts
