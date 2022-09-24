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
        window_length,
        epochs=1,
        buffer_size=100000,
        context_window_type="double",
    ):
        self.window_length = window_length
        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None
        self.seqs = seqs
        self.epochs = epochs
        self.context_window_type = context_window_type

        # Count sequence elements
        counter = Counter()
        self.n_samples = 0
        for seq in seqs:
            counter.update(seq)
            n_pairs = count_center_context_pairs(
                window_length, len(seq), context_window_type
            )
            self.n_samples += n_pairs
        self.n_elements = int(max(counter.keys()) + 1)
        self.ele_null_prob = np.zeros(self.n_elements)
        for k, v in counter.items():
            self.ele_null_prob[k] = v
        self.ele_null_prob /= np.sum(self.ele_null_prob)
        self.n_seqs = len(seqs)
        self.seq_order = np.random.choice(self.n_seqs, self.n_seqs, replace=False)
        self.seq_iter = 0

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.n_samples * self.epochs

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id]
        cont = self.contexts[self.sample_id].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id].astype(np.int64)

        self.sample_id += 1

        return center, cont, rand_cont

    def _generate_samples(self):
        self.centers = []
        self.contexts = []
        for i in range(self.buffer_size):
            self.seq_iter += 1
            if self.seq_iter >= self.n_seqs:
                self.seq_iter = self.seq_iter % self.n_seqs
            seq_id = self.seq_order[self.seq_iter]
            cent, cont = _get_center_context_pairs(
                np.array(self.seqs[seq_id]),
                self.window_length,
                self.context_window_type,
            )
            self.centers.append(cent)
            self.contexts.append(cont)
        self.centers, self.contexts = (
            np.concatenate(self.centers),
            np.concatenate(self.contexts),
        )

        self.random_contexts = np.random.choice(
            self.n_elements, size=len(self.centers), p=self.ele_null_prob, replace=True
        )
        self.n_sampled = len(self.centers)
        self.sample_id = 0


@njit(nogil=True)
def count_center_context_pairs(window_length, seq_len, context_window_type):
    # Count the number of center-context pairs.
    # Suppose that we sample, for each center word, a context word that proceeds the center k-words.
    # There are T-k words in the sequence, so that the number of pairs is given by summing this over k upto min(T-1, L), i.e.,
    # 2 \sum_{k=1}^{min(T-1, L)} (T-k)
    # where we cap the upper range to T-1 in case that the window covers the entire sentence, and
    # we double the sum because the word window extends over both proceeding and succeeding words.
    min_window_length = np.minimum(window_length, seq_len - 1)
    n = 2 * min_window_length * seq_len - min_window_length * (min_window_length + 1)
    if context_window_type == "double":
        return int(n)
    else:
        return int(n / 2)


@njit(nogil=True)
def _get_center_context_pairs(seq, window_length, context_window_type):
    """Get center-context pairs from sequence.
    :param seq: Sequence
    :type seq: numpy array
    :param window_length: Length of the context window
    :type window_length: int
    :return: Center, context pairs
    :rtype: tuple
    """
    n_seq = len(seq)
    n_pairs = count_center_context_pairs(window_length, n_seq, context_window_type)
    centers = -np.ones(n_pairs, dtype=np.int64)
    contexts = -np.ones(n_pairs, dtype=np.int64)
    idx = 0
    wstart, wend = 0, 2 * window_length + 1
    if context_window_type == "suc":
        wstart = window_length + 1
    if context_window_type == "prec":
        wend = window_length

    for i in range(n_seq):
        for j in range(wstart, wend):
            if (
                (j != window_length)
                and (i - window_length + j >= 0)
                and (i - window_length + j < n_seq)
            ):
                centers[idx] = seq[i]
                contexts[idx] = seq[i - window_length + j]
                idx += 1
    return centers, contexts