#!/bin/bash -v

for nfillers in 100 1000 500 750 250
do
  for fillerdim in 100 1000 500 750 250
  do
    if  [ $nfillers -eq 250 ]; then
      python experiments.py word2vec --cuda 0 --n_fillers ${nfillers} --filler_dim ${fillerdim} --topk 1 5 10 25
    fi 
    # python experiments.py type1 --cuda 0 --n_fillers ${nfillers} --filler_dim ${fillerdim} 
    # python experiments.py type2 --cuda 0 --n_fillers ${nfillers} --filler_dim ${fillerdim}
  done
done
