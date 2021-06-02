#!/bin/bash

# flag re: should we combine multi-token entities into single token spans or not? options: ent_tokens, no_ent_tokens
declare -a combineEntTokens=("True")

# flag re: should we use the premise and hypothesis (True) or just the hypothesis (False)
declare -a usePremise=("True" "False")

# We only compute PMI for the training dataset
declare -a splits=("train")
#

# Set flags for which actions we want to run
pmi=true
hypLen=true

source activate clinical_nli

if [ "$pmi" = true ] ; then
for i in "${splits[@]}"; do
   for j in "${combineEntTokens[@]}"; do
      echo "Computing and ranking token-class PMI by label (for hypotheses) for split: ${i} and combine ents flag: ${j}"
      python3 ./../src/analysis/lexical.py --config_file="./../example_cfg.ini" --split="$i" --combine_ents="$j"
    done
  done
fi

