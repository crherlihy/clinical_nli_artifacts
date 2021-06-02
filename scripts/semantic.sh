#!/bin/bash
#declare -a splits=("train" "dev" "test")
declare -a splits=("all")

declare -a segments=("premise" "hypothesis")
# declare -a segments=("premise" "hypothesis")

# flag re: should we combine multi-token entities into single token spans or not? options: ent_tokens, no_ent_tokens
declare -a combineEntTokens=("False")

linkTo=mesh

# Set flags for which actions we want to run
meshLinks=true

source activate clinical_nli

if [ "$meshLinks" = true ] ; then
for i in "${splits[@]}"; do
   for j in "${segments[@]}"; do
     for k in "${combineEntTokens[@]}"; do
        echo "[INFO]: Getting ${linkTo} linked ents + metadata | SPLIT: ${i} | SEGMENT: ${j} | COMBINE_ENTSs: ${k}"
        python3 ./../src/analysis/semantic.py --config_file="./../example_cfg.ini" --splits="$i" --segment="$j" --combine_ents="$k" --link_to="$linkTo"
      done
    done
  done
fi

