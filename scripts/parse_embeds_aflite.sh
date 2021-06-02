#!/bin/bash

# flag re: should we combine multi-token entities into single token spans or not? options: ent_tokens, no_ent_tokens
#declare -a combineEntTokens=("True" "False")
#declare -a combineEntTokens=("False")
declare -a combineEntTokens=("True")

# flag re: should we use the premise and hypothesis (True) or just the hypothesis (False)
#declare -a usePremise=("True")
declare -a usePremise=("True" "False")

# train/dev/test splits of the complete/original MedNLI dataset
#declare -a splits=("train" "dev" "test")
declare -a embeds_splits=("all")


# Set flags for which actions we want to run
fastText=true
ftAllSubsets=true
embeddings=true
aflite=true

source  activate clinical_nli

# create fastText version of the JSON train/dev/test files (e.g., __label__{} premise hypothesis OR __label__{} hypothesis)
if [ "$fastText" = true ] ; then
for i in "${splits[@]}"; do
   for j in "${usePremise[@]}"; do
      echo "Creating fastText format file for split: ${i} and use_premise flag: ${j}"
      python3 ./../src/utils/parse_input_data.py  --config_file="./../example_cfg.ini" --split="$i" --incl_premise="$j"
    done
  done
fi

if [ "$ftAllSubsets"  = true ] ; then
  for j in "${usePremise[@]}"; do
    python3 ./../src/utils/create_joint_txt.py --config_file="./../example_cfg.ini"  --incl_premise="$j"
  done
fi

# create/recover embeddings
if [ "$embeddings" = true ] ; then
for i in "${embeds_splits[@]}"; do
   for j in "${usePremise[@]}"; do
      for k in "${combineEntTokens[@]}"; do
        echo "Creating embeddings for split: ${i} and use_premise flag: ${j} and combineEnts flag: ${k}"
        python3  ./../src/utils/get_embeddings.py  --config_file="./../example_cfg.ini" --embeds_splits="$i" --incl_premise="$j" --combine_ents="$k"
      done
    done
  done
fi

# run AFLite (adversarial filtering algorithm) to partition X_train into  X_train_easy and X_train_hard
if [ "$aflite" = true ] ; then
for i in "${embeds_splits[@]}"; do
   for j in "${usePremise[@]}"; do
      for k in "${combineEntTokens[@]}"; do
        echo "Running  AFLite to create easy/hard subsets for split: ${i} and use_premise flag: ${j} and combineEnts flag: ${k}"
        python3 ./../src/filter/filter.py  --config_file="./../example_cfg.ini" --embeds_splits="$i" --incl_premise="$j" --combine_ents="$k"
      done
    done
  done
fi


