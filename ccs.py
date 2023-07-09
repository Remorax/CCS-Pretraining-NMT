import textdistance, random, sys, re, time, pickle
from collections import defaultdict
from unicodedata import category
import functools
import operator
import numpy as np
import argparse
import unicodedata as ud
from collections import Counter
from ordered_set import OrderedSet

def ischar(word):
	return len([c for c in word if not c.isalpha() and not c.isdigit() and c.strip()]) == len(word)

def is_clean(src_sent, tgt_sent, src_lang, tgt_lang):

	if len(src_sent) * len(tgt_sent):
		a = len(src_sent.split())
		b = len(tgt_sent.split())
		if min(a/b, b/a) < args.min_parallel_length_ratio:
			return False

	if src_sent == tgt_sent:
		return False

	return True

def extract_phrase_alignments(source_sent_words, translated_sent_words, alignment):

	src_to_translated, translated_to_src = defaultdict(set), defaultdict(set)
	for alignment_pair in alignment.split():
		s,t = alignment_pair.split("-")
		src_to_translated[int(s)].add(int(t))
		translated_to_src[int(t)].add(int(s))

	visited = {idx: False for idx,word in enumerate(source_sent_words)}
	all_choices = []
	for src_idx, src_word in enumerate(source_sent_words):
		if visited[src_idx]:
			continue
		if src_idx not in src_to_translated:
			visited[src_idx] = True
			continue
		prev_src_words, prev_target_words = [], []
		source_words, target_words = {src_idx}, set()
		while True:
			for sw in source_words:
				target_words = target_words.union(src_to_translated[sw])
			if len(source_words) == len(prev_src_words):
				if source_words == prev_src_words:
					break
			if len(target_words) == len(prev_target_words):
				if target_words == prev_target_words:
					break
			prev_target_words = set(target_words)
			prev_src_words = set(source_words)
			for tw in target_words:
				source_words = source_words.union(translated_to_src[tw])
		for idx in source_words:
			visited[idx] = True
		if source_words and target_words:
			if not any([source_sent_words[word].lower().isalpha() for word in source_words]):
				continue
			if args.force_1to1:
				if len(source_words) > 1 or len(target_words) > 1:
					continue
			all_choices.append((sorted(list(source_words)), sorted(list(target_words))))
	all_choices = sorted(all_choices, key=lambda el:el[0][0])
	return all_choices

def conduct_cas(src_line, translations):
	global src_lens, tgt_lens, indexes
	all_langs_choices = []
	source_sent_words = src_line.split()
	translated_sents_words = [translated_line.split() for (translated_line, _, _) in translations]

	for i,(translated_line, alignment, lang_idx) in enumerate(translations):
		all_langs_choices.append((i, lang_idx, extract_phrase_alignments(source_sent_words, translated_sents_words[i], alignment)))
	if args.blcs:
		all_langs_choices = [random.choice(all_langs_choices)]
	visited = {idx: True if word.isdigit() or ischar(word) else False for idx,word in enumerate(src_line.split())}
	replacements = []
	replaced_count = 0
	while True:
		idx, lang_idx, curr_choices = random.choice(all_langs_choices)
		possible_choices = [(src_idxs, translated_idxs) for src_idxs, translated_idxs in curr_choices if not any([visited[src_idx] for src_idx in src_idxs]) ]
		if not possible_choices:
			break
		if float(replaced_count/len(source_sent_words)) > args.replacement_ratio:
			break
		src_idxs, translated_idxs = random.choice(possible_choices)
		start_rep_idx = src_idxs[0]
		visited[start_rep_idx] = True
		end_rep_idx = int(start_rep_idx)
		src_idxs_filt = [start_rep_idx]
		replaced_count += 1
		
		for w in sorted(src_idxs[1:]):
			replaced_count += 1
			end_rep_idx = w
			src_idxs_filt.append(w)
			visited[w] = True
		replacements.append((idx, lang_idx, (src_idxs_filt, translated_idxs)))
	replacements = sorted(replacements, key=lambda l:l[2][0])

	new_sentence, prev_start_idx, end_rep_idx = [], 0, len(source_sent_words)
	visited = [False for _ in source_sent_words]
	
	for (idx, lang_idx, (src_idxs, translated_idxs)) in replacements:
		start_rep_idx, end_rep_idx = src_idxs[0], src_idxs[-1]
		indexes[lang_idx] += 1
		translated_words = [str(translated_sents_words[idx][w]) for w in translated_idxs]
		new_sentence.extend(source_sent_words[prev_start_idx: start_rep_idx])
		new_sentence.extend(translated_words)
		src_lens.append(end_rep_idx + 1 - start_rep_idx)
		tgt_lens.append(len(translated_words))
		prev_start_idx = end_rep_idx+1

	new_sentence.extend(source_sent_words[prev_start_idx:])

	new_sentence = " ".join(new_sentence)

	return new_sentence

parser = argparse.ArgumentParser()

parser.add_argument("--source_file", help="Provide path to source file to CAS. Ensure file is suffixed by its language code eg. train.en or train.fr")
parser.add_argument("--reference_file", help="Provide path to reference file used for training model", type=str, nargs="?", default='')
parser.add_argument("--translation_files", help="Provide path to file(s) containing generated translations. If there are multiple files (multiple translations), separate the files by commas.  Ensure file is suffixed by its language code eg. translated.en or translated.fr")
parser.add_argument("--alignment_files", help="Provide path to alignments between source and translated file(s).  If there are multiple files (multiple translations), separate the corresponding alignments by commas.")
parser.add_argument("--output_prefix", help="Provide prefix for storing final CASed source and target files")

parser.add_argument("--replacement_ratio", help="Provide ratio with which to conduct randomized replacement", type=float, default=0.6)
parser.add_argument("--random_seed", help="Provide seed for randomization", type=int, default=0)
parser.add_argument("--blcs", help="Whether to use Bilingual Code-Switching (BLCS) or not", action="store_true")
parser.add_argument("--force_1to1", help="Whether or not to force 1:1 replacement", action="store_true", default=False)
parser.add_argument("--min_parallel_length_ratio", help="Provide max length ratio between parallel sentences (for cleaning)", type=float, default=0.6)

args = parser.parse_args()
random.seed(int(args.random_seed))
np.random.seed(int(args.random_seed))

src_lang = args.source_file.strip().split(".")[-1]
translated_langs = [file.strip().split(".")[-1] for file in args.translation_files.split(",")]

if args.reference_file:
	reference_file = open(args.reference_file)
	ref_lang = args.reference_file.strip().split(".")[-1]

pointers = [(open(file[0].strip()), open(file[1].strip())) for file in zip(args.alignment_files.split(","),args.translation_files.split(","))]
CASed_sentences, target_sentences, alignments_new, translations_new = [], [], [], []
src_lens, tgt_lens = [], []

indexes = [0 for file in args.translation_files.split(",")]

src_count = 0
source_file=open(args.source_file)

for src_sent in source_file:

	possible_translations = [(next(ptr[0]).strip(), next(ptr[1]).strip()) for ptr in pointers]
	candidates = [(translation[1], translation[0], i) for i,translation in enumerate(possible_translations) if translation[0].strip() and is_clean(src_sent, translation[1], src_lang, translated_langs[i])]
	if args.reference_file:
		ref_sent = next(reference_file).strip()

	if not candidates:
		continue

	CASed_sentence = conduct_cas(src_sent, candidates)
	CASed_sentences.append(CASed_sentence)
	if args.reference_file:
		target_sentences.append(ref_sent)

	if src_count % 1000000 == 0:
		print (f"Sample sentence: {src_sent} CASed sentence: {CASed_sentence}")
	src_count += 1

print (f"Total file length (uncleaned): {len(CASed_sentences)} vs cleaned: {src_count}")

print ("All done")
print ("Mean source span replacement length: ", np.mean(src_lens))
print ("Mean target span replacement length: ", np.mean(tgt_lens))
print ("Number of replacements per language: ", list(zip(translated_langs,indexes)))
open(args.output_prefix + f".{src_lang}", "w+").write("\n".join(CASed_sentences))
if args.reference_file:
	open(args.output_prefix + f".{ref_lang}", "w+").write("\n".join(target_sentences))

