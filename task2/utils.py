

def process_bio(tokens, subtokens, bios, bpe_indicator='Ġ'):
    starts = [0]
    starts += list(filter(lambda j: subtokens[j][0] == bpe_indicator, range(len(subtokens))))
    assert len(starts) == len(tokens)
    assert len(tokens) == len(bios)
    input_mask = [1] * len(subtokens)
    outbio = [-1] * len(subtokens)
    bio_mask = [0] * len(subtokens)
    for i, st in enumerate(starts):
        outbio[st] = bios[i]
        bio_mask[st] = 1
    return input_mask, outbio, bio_mask


def get_start_idxs_batched(b_tokens, b_sub_tokens, block_size, bpe_indicator='Ġ'):
    from copy import deepcopy as cp
    ss = []
    es = []
    batch_size = len(b_tokens)
    for i in range(batch_size):
        starts = [1]
        starts += list(filter(lambda j: b_sub_tokens[i][j][0] == bpe_indicator, range(len(b_sub_tokens[i]))))
        assert len(starts) == len(b_tokens[i])
        ends = cp(starts[1:])
        ends.append(len(b_sub_tokens[i]) - 1)	# should be right open interval, doesnot matter here as SEP
        ends = list(filter(lambda x: x <= block_size - 2, ends))       # less than 512, remove CLS SEP
        ss.append(starts[:len(ends)])
        es.append(ends)
    return ss, es
