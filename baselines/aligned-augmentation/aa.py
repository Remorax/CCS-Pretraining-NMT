import glob, sys, random, argparse, os
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--source_file", help="Provide path to source file to be RASed")
parser.add_argument("--output_file", help="Provide path to final (RASed) output file")
parser.add_argument("--muse_dir", help="Provide path to directory containing MUSE dicts in {src}-{tgt}.txt format")

parser.add_argument("--replacement_ratio", help="Provide ratio with which to conduct randomized replacement", type=float, default=0.63)
parser.add_argument("--langs", help="Provide MUSE dictionary languages to sample")
parser.add_argument("--is_parallel", help="Declare if corpus to be replaced is parallel (True) or monolingual (False)", action="store_true")
parser.add_argument("--random_seed", help="Provide seed for randomization", type=int, default=0)

args = parser.parse_args()
random.seed(args.random_seed)

src = args.source_file.split(".")[-1].split("_")[0]
muse_dicts = {}
for tgt in args.langs.split(","):
    muse_dict_path = f"{args.muse_dir}/{src}-{tgt}.txt"
    muse_dict = defaultdict(list)
    if not os.path.exists(muse_dict_path):
        print (muse_dict_path, "does not exist")
        continue
    for l in open(muse_dict_path).read().split("\n"):
        if l.strip():
            try:
                muse_dict[l.strip().split(" ")[0]].append(l.strip().split(" ")[1])
            except:
                muse_dict[l.strip().split("\t")[0]].append(l.strip().split("\t")[1])
    muse_dicts[tgt] = muse_dict

src_sents = open(args.source_file, "r")
final_sents = []
total_words = 0
replacement_stats = {}
for j,src_sent in enumerate(src_sents):
    src_sent = src_sent.strip()
    if not src_sent:
        continue

    src_words = src_sent.split()
    src_words_replace = list(src_words)
    if args.is_parallel:
        lang, muse_dict = random.choice(list(muse_dicts.items()))
    for i,word in enumerate(src_words):
        total_words += 1 
        if not args.is_parallel:
            lang, muse_dict = random.choice(list(muse_dicts.items()))
        if word in muse_dict:
            if random.random() <= args.replacement_ratio:
                if lang not in replacement_stats:
                    replacement_stats[lang] = 1
                else:
                    replacement_stats[lang] += 1
                translation = random.choice(muse_dict[word])
                tp = src_words_replace[:i] + [translation] + src_words_replace[i+1:]
                src_words_replace = tp
    sent = " ".join(src_words_replace)
    if j%100000==0:
        print (f"Sample sentence: {sent}")
    final_sents.append(sent)

print (f"For {args.source_file} replacement stats: {replacement_stats} total_words: {total_words}")
open(args.output_file,"w+").write("\n".join(final_sents) + "\n")
