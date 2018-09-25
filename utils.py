def to_np(x):
    return x.data.cpu().numpy()


def trim_seqs(seqs):
    """

    :param seqs:
    :return:
    """
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for idx in to_np(output_seq):
            trimmed_seq.append(idx[0])
            if idx == 2:
                break
        trimmed_seqs.append(trimmed_seq)
    return trimmed_seqs


def seq_to_string(seq, idx_to_tok, input_tokens=None):
    """

    :param seq:
    :param idx_to_tok:
    :param input_tokens:
    :return:
    """
    vocab_size = len(idx_to_tok)
    seq_length = (seq != 0).sum()
    words = []
    for idx in seq[:seq_length]:
        if idx < vocab_size:
            words.append(idx_to_tok[idx])
        elif input_tokens is not None:
            words.append(input_tokens[idx - vocab_size])
        else:
            words.append('<???>')
    string = ''.join(words)
    return string