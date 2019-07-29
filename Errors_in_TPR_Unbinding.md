# Errors in TPR Unbinding

---

#### Paul Smolensky and Coleman Haley

In this notebook, we provide theoretical and experimental
results on the error rate of TPR binding. We show that
this error is bounded in a theoretical worst case, that
this theoretical error is experimentally validated, and
that empirically error appears to be lower in cases other
than the theoretical worse case. We consider experiments
for empirical error in the case of totally random
distribution of bound vectors, as well as distributions
from real-world applications of TPRs in the domain of
natural language processing. We show that the error
increases nicely (:P) as a function of the ratio of filler
vectors to role vectors.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
```

```python

def torch_sample_spherical(npoints, ndim, nsamples, device):
   vecs = torch.randn(ndim, npoints, nsamples).to(device)
   vecs /= torch.norm(vecs, dim=0)
   return vecs

```

```python
def type1_experiment(device):
    print("Beginning Type 1 experiments...")
    nfillers = 50
    print(f"Number of fillers ${nfillers}")
    fillerdim = 200
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
    for nbindings in tqdm(range(1, 1000, 100)):
        rolevecs = torch_sample_spherical(
            npoints=nbindings + 1, ndim=roledim, nsamples=nsamples, device=device
        )

        other = torch.randint(0, nfillers, (nsamples,)).to(device)
        bindings = other.repeat(nbindings, 1).to(device)
        bindings[0, :] = torch.randint(0, nfillers, (nsamples,)).to(device)
        fillervecs = torch.gather(
            fillerbank, 1, bindings.unsqueeze(0).expand(fillerdim, nbindings, nsamples)
        ).to(device)
        print(fillerbank)

        T = torch.einsum("fbs, rbs -> frs", fillervecs, rolevecs[:, :nbindings, :]).to(device)
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

    filename = "type1_results.npy"
    print(f"Writing the estimated probabilities to ./${filename}")
    np.save(filename, error_prob)
type1_experiment(torch.device('cpu'))
```

```python
def type2_experiment(device):
    y1 = np.zeros(999)
    n = 500
    nsamples = 2000
    nfillers=50
    fillerdim=200
    N=50
    F = vec = torch_sample_spherical(npoints=nfillers, ndim=fillerdim, nsamples=1, device=device).expand(fillerdim, nfillers, nsamples) # zipf distribution?
    for k in tqdm(range(600,1000,10)):
      rhats = torch_sample_spherical(npoints=k+1, ndim=n, nsamples=nsamples, device=device)
      filleris = torch.randint(0, N, (k, nsamples)).to(device).unsqueeze(0).expand(fillerdim, k, nsamples) #type 2 error
      fillers = torch.gather(F, 1, filleris).to(device)
      T = torch.einsum('ifs, jfs -> ijs', fillers, rhats[:,:k,:]).to(device)
      A_hats = filleris[0, 0, :nsamples] # correct filler label of r_0
      f_tilde = torch.einsum('frs, rs -> fs', T, rhats[:, 0, :nsamples]).to(device)
      distances = torch.einsum('fns, fs -> ns', F, f_tilde).to(device)
      closest = torch.argmax(distances, dim=0).to(device)
      errors = (A_hats - closest).nonzero()
      print(len(errors))
      y1[k-1] = len(errors) / 2000
      print(y1[k-1])
      del errors
      del closest
      del distances
      del f_tilde
      del T
      del fillers
      del rhats
      del filleris
    filename = 'type2_results.npy'
    np.save(filename, y1)
    return y1
results = type2_experiment(torch.device('cpu'))

```

```python
x1 = np.array(range(1,1000))
y1 = np.zeros(999)
n = 500
nsamples = 2000
nfillers=50
fillerdim=200
N=50
F = vec = torch_sample_spherical(npoints=nfillers, ndim=fillerdim, nsamples=1).expand(fillerdim, nfillers, nsamples) # zipf distribution?
for k in range(1,1000,5):
  error_c = 0
  rhats = torch_sample_spherical(npoints=k+1, ndim=n, nsamples=nsamples)
  filleris = torch.randint(0, N, (k, nsamples)).unsqueeze(0).expand(fillerdim, k, nsamples) #type 2 error
  # print(filleris[0,0,:2000])
  #fillers = torch.einsum()
  fillers = torch.gather(F, 1, filleris)
  # print(torch.allclose(fillers[:,0,0],F[:,filleris[0,0,0].data,0]))
  T = torch.einsum('ifs, jfs -> ijs', fillers, rhats[:,:k,:])
  #print(T[0,0,0])
  #print(sum(torch.ger(fillers[:,i,0],  rhats[:,i,0]) for i in range(k)))
  A_hats = filleris[0, 0, :2000] # correct filler label of r_0
  f_tilde = torch.einsum('frs, rs -> fs', T, rhats[:, 0, :nsamples])
  # print(T[:,:,0] @ rhats[:,0,0])
  # print('f tilde!')
  # print(f_tilde[0,0])
  # print(F[:,filleris[0,0,0].data, 0])
  distances = torch.einsum('fns, fs -> ns', F, f_tilde)
  closest = torch.argmax(distances, dim=0)
  errors = (A_hats - closest).nonzero()
  # print(len(errors))
  y1[k-1] = len(errors) / 2000
  print(y1[k-1])
  if k > 200:
    break
#     T =
#         T = sum([np.outer(F[:,fillers[i]], rhats[:,i]) for i in range(k)]) # may choose non-distinct vectors
#         A_hat = F[:, fillers[0]]

#         f_tilde = np.inner(T, rhats[:,0])
#         closest = max(F.T, key=lambda f:np.inner(f_tilde, f))
#         if fillers[0]!=np.where(F.T==closest)[0][0]:
#             error_c += 1
#     y1[k-1] = error_c / 250
```
