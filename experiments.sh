#!/bin/bash -v

for nfillers in 100 1000 500 750 250
do
  for fillerdim in 100 1000 500 750 250
  do
    if  [nfillers = 100]; then
      python experiments.py word2vec --cuda --nfillers ${nfillers} --filler_dim ${fillerdim} &
    fi &
    python experiments.py type1 --cuda --n_fillers ${nfillers} --filler_dim ${fillerdim} &
    python experiments.py type2 --cuda --n_fillers ${nfillers} --filler_dim ${fillerdim}
  end
end
