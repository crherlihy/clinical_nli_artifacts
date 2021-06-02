import os
import configparser
import argparse

def create_mednli_all_file(data_dir: str, splits: [str], combined_nli_file: str, lower: bool, incl_premise: bool, separator:str="ñ"):
    combined_file = open(combined_nli_file, "w")

    sep_str = "sep" if separator != "None" else ""

    for s in splits:

        s_file = os.path.join(data_dir, "mli_{}_{}_v1_{}.txt".format(s, "w_premise" if incl_premise else "no_premise", sep_str))

        with open(s_file, 'r') as f:
            for i,line in enumerate(f.readlines()):
                combined_file.write(line)
        f.close()
    combined_file.close()
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parse MedNLI JSON files and create a single joint txt file for fastText model')
    parser.add_argument('--config_file', type=str, default='./../../example_cfg.ini', nargs='?', const='./example_cfg.ini')
    parser.add_argument('--incl_premise', type=str, default="False", nargs='?', const="False")
    parser.add_argument('--separator', type=str, default="ñ", nargs='?')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    FT_DIR = config.get("paths", "FT_DIR")

    INCL_PREMISE = eval(args.incl_premise)
    SPLITS = ["train", "dev", "test"]
    PREMISE_str = "w_premise" if INCL_PREMISE else "no_premise"
    SEP_str = "sep" if args.separator != "None" else ""

    print(os.getcwd())

    if not os.path.isdir(FT_DIR):
        os.mkdir(FT_DIR)

    combined_file = os.path.join(FT_DIR, "mli_all_{}_v1_{}.txt".format(PREMISE_str, SEP_str))
    create_mednli_all_file(data_dir=FT_DIR, splits=SPLITS, combined_nli_file=combined_file, lower=True,
                             incl_premise=INCL_PREMISE, separator=args.separator)