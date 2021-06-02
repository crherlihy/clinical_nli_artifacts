#!/bin/bash

# flag re: should we combine multi-token entities into single token spans or not? options: ent_tokens, no_ent_tokens
declare -a combineEntTokens=("True") # set to False if running fastText baseline; set to True if computing results for AfLite partitions

declare -a inclPremise=("w_premise" "no_premise")

# Set flags for which actions we want to run
evalAflite="True" # set to "False" if running fastText baseline; set to "True" if computing results for AfLite partitions (AfLite must be run first)

source activate clinical_nli

# Train and evaluate a baseline fastText classifier for each hyperparmeter combination
for i in "${combineEntTokens[@]}"; do
  for j in "${inclPremise[@]}"; do
      echo "Computing fastText baselines | combine_ents flag: ${i}"
      python3 ./../src/models/fasttext_baseline.py --config_file="./../example_cfg.ini" --combine_ents="$i" --incl_premise="$j" --eval_aflite="$evalAflite"
    done
done
