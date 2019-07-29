import argparse
import os
#from tqdm import tqdm

import numpy as np
import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", nargs="?", type=str, default="")
    parser.add_argument("--cuda", nargs="?", const="-1", type=int)
    parser.add_argument("--n_fillers", type=int)
    parser.add_argument("--filler_dim", type=int)
    parser.add_argument("--topk", nargs="*", type=int, default=[1])
    return parser.parse_args()


def torch_sample_spherical(npoints, ndim, nsamples, device):
    vecs = torch.randn(ndim, npoints, nsamples).to(device)
    vecs /= torch.norm(vecs, dim=0)
    return vecs

def get_freer_gpu():
    import os

    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >cuda")
    memory_available = [int(x.split()[2]) for x in open("cuda", "r").readlines()]

    return np.argmax(memory_available)


def type1_experiment(nfillers, fillerdim, device):
    print("Beginning Type 1 experiments...")
    print(f"Number of fillers ${nfillers}")
    print(f"Dimensionality of fillers ${fillerdim}")
    nsamples = 250
    print(f"Number of samples to estimate probability")
    roledim = 500

    error_prob = np.zeros(1000)

    print("Generating filler vectors")
    fillerbank = torch_sample_spherical(
        npoints=nfillers, ndim=fillerdim, nsamples=1, device=device
    ).expand(
        fillerdim, nfillers, nsamples
    )  # zipf distribution?
    print("Beginning error probability estimation")
    for nbindings in range(1, 1000, 5):
        rolevecs = torch_sample_spherical(
            npoints=nbindings + 1, ndim=roledim, nsamples=nsamples, device=device
        )

        other = torch.randint(0, nfillers, (nsamples,)).to(device)
        bindings = other.repeat(nbindings, 1).to(device)
        bindings[0, :] = torch.randint(0, nfillers, (nsamples,)).to(device)
        fillervecs = torch.gather(
            fillerbank, 1, bindings.unsqueeze(0).expand(fillerdim, nbindings, nsamples)
        ).to(device)

        T = torch.einsum("fbs, rbs -> frs", fillervecs, rolevecs[:, :nbindings, :]).to(
            device
        )
        role_zero_filler = bindings[0, :nsamples]  # correct filler
        role_zero_unbinding = torch.einsum(
            "frs, rs -> fs", T, rolevecs[:, 0, :nsamples]
        ).to(device)

        distances = torch.einsum("fns, fs -> ns", fillerbank, role_zero_unbinding).to(
            device
        )
        closest = torch.argmax(distances, dim=0)
        errors = (role_zero_filler - closest).nonzero()
        error_prob[nbindings - 1] = len(errors) / nsamples
        del rolevecs
        del bindings
        del T
        del role_zero_unbinding
        del distances
        del closest
        del errors
    filename = f"results/type1_results_nfillers_{nfillers}_fillerdim_{fillerdim}.npy"
    np.save(filename, error_prob)


def type2_experiment(nfillers, fillerdim, device):
    x1 = np.array(range(1, 1000))
    error_prob = np.zeros(999)
    n = 500
    nsamples = 250
    N = 50
    F = vec = torch_sample_spherical(
        npoints=nfillers, ndim=fillerdim, nsamples=1, device=device
    ).expand(
        fillerdim, nfillers, nsamples
    )  # zipf distribution?
    for k in range(1, 1000, 5):
        rhats = torch_sample_spherical(
            npoints=k + 1, ndim=n, nsamples=nsamples, device=device
        )
        filleris = (
            torch.randint(0, N, (k, nsamples))
            .to(device)
            .unsqueeze(0)
            .expand(fillerdim, k, nsamples)
        )  # type 2 error
        fillers = torch.gather(F, 1, filleris).to(device)
        T = torch.einsum("ifs, jfs -> ijs", fillers, rhats[:, :k, :]).to(device)
        A_hats = filleris[0, 0, :nsamples]  # correct filler label of r_0
        f_tilde = torch.einsum("frs, rs -> fs", T, rhats[:, 0, :nsamples]).to(device)
        distances = torch.einsum("fns, fs -> ns", F, f_tilde).to(device)
        closest = torch.argmax(distances, dim=0).to(device)
        errors = (A_hats - closest).nonzero()
        error_prob[k - 1] = len(errors) / nsamples
        del errors
        del closest
        del distances
        del f_tilde
        del T
        del A_hats
        del fillers
        del rhats
        del filleris
    filename = f"results/type2_results_nfillers_{nfillers}_fillerdim_{fillerdim}.npy"
    np.save(filename, error_prob)


def word2vec_experiment(fillerdim, topk, device):
    print("Downloading corpus and nltk tools, if necessary.")
    import nltk
    import gensim
    from gensim.models import KeyedVectors

    nltk.download("reuters")
    nltk.download("punkt")

    print("Beginning preprocessing...")
    from nltk.corpus import reuters

    sents = list(reuters.sents())
    print("Getting sentences of length 50 or less.")
    sents.sort(key=len)
    bound = 0
    while len(sents[bound + 1]) <= 50:
        bound += 1
    # constrain upper length of sentences to have a fixed role set.
    sents = sents[:bound]  # the sentences of length 50 or less

    # print('Removing sentences of length 0.')
    len0sents = []
    for i, sent in enumerate(sents):
        if len(sent) == 0:
            len0sents.append(sent)
        else:
            sents[i] = [word.lower() for word in sent]
    for sent in len0sents:
        sents.remove(sent)
    print("%d sentences." % (len(sents)))
    num_sents_per_len = [
        sum(1 if len(sent) == i else 0 for sent in sents) for i in range(1, 50)
    ]
    print(num_sents_per_len)

    try:
        wv = KeyedVectors.load("reuters.wv", mmap="r")
        print("Pretrained wordvectors loaded.")
    except:
        print("Training word2vec model.")
        model = gensim.models.Word2Vec(sents, size=fillerdim, min_count=1, workers=4)
        print("Word2Vec training complete!")
        model.wv.save("reuters.wv")
        wv = model.wv

    print("Beginning experiment...")
    error_prob = np.zeros(len(topk), 49)
    # error_prob_topk = np.zeros(49)
    nsamples = 1
    sents.sort(key=len)
    breaks = [i for i in range(1, len(sents)) if len(sents[i]) != len(sents[i - 1])]
    fillerbank = torch.from_numpy(wv.vectors.T)
    roledim = 100
    for i, brk in enumerate(breaks):
        rhats = torch_sample_spherical(
            npoints=50, ndim=roledim, nsamples=nsamples, device=device
        )
        if i == 0:
            to_process = sents[:brk]
        else:
            to_process = sents[breaks[i - 1] : brk]
        nbindings = len(to_process[0])
        sents.sort(key=len)
        print("stacking sentence vectors")
        fillervecs = (
            torch.stack(
                [
                    torch.stack(
                        [torch.from_numpy(wv[sent[j]]) for j in range(nbindings)], dim=1
                    )
                    for sent in to_process
                ],
                dim=2,
            )
            .to(device)
            .unsqueeze(-1)
            .expand(-1, -1, -1, nsamples)
        )
        rolevecs = (
            rhats[:, :nbindings, :].unsqueeze(2).expand(-1, -1, len(to_process), -1)
        )
        print("binding...")
        T = torch.einsum("fbns, rbns -> frns", fillervecs, rolevecs).to(device)
        role_unbinding = torch.einsum("frns, rbns -> fbns", T, rolevecs).to(device)
        print("calc dist...")
        distances = torch.einsum(
            "fvbns, fbns -> vbns",
            fillerbank.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(2)
            .expand(-1, -1, nbindings, len(to_process), nsamples),
            role_unbinding,
        ).to(device)
        print("yayyyy")
        for ki, k in enumerate(topk):
            tops = torch.topk(distances, k, dim=0)
            intopk = torch.min(
                torch.abs(
                    torch.sum(
                        topkvecs - fillervecs.unsqueeze(1).expand(-1, k, -1, -1, -1),
                        dim=0,
                    )
                ),
                dim=0,
            )
            error[ki][i] = len(intopk.values.nonzero()) / intopk.values.numel()
        # filename = "tops_word2vec_len" + str(i) + ".pt"
        # torch.save(tops, filename)
    filename = f'results/word2vec_results_fillerdim_{fillerdim}_topk_{topk.replace(" ", "")}.pt'
    torch.save(error, filename)


def main():
    args = parse_arguments()
    device = torch.device("cpu")
    try:
        if not torch.cuda.is_available():
            print("Cuda is not available. Using cpu instead.")
            raise AttributeError()
        device_num = args.cuda if args.cuda != -1 else get_freer_gpu()
        device = torch.device(f"cuda:{device_num}")
        print(f"Using cuda:${device_num} as the torch device.")
    except AttributeError:
        pass

    if args.experiment == "type1":
        type1_experiment(args.n_fillers, args.filler_dim, device)
    elif args.experiment == "type2":
        type2_experiment(args.n_fillers, args.filler_dim, device)
    elif args.experiment == "word2vec":
        word2vec_experiment(args.filler_dim, args.topk, device)
    else:
        type1_experiment(args.n_fillers, args.filler_dim, device)
        type2_experiment(args.n_fillers, args.filler_dim, device)
        word2vec_experiment(args.filler_dim, args.topk, device)


if __name__ == "__main__":
    main()
