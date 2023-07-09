# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    iterators,
    FairseqDataset,
    LanguagePairDataset)
from ..data import SubsampleLanguagePairDataset

import logging
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

logger = logging.getLogger(__name__)


def concat_language_pair_dataset(*language_pair_datasets, up_sample_ratio=None,
                                 all_dataset_upsample_ratio=None):
    logger.info("To cancat the language pairs")
    dataset_number = len(language_pair_datasets)
    if dataset_number == 1:
        return language_pair_datasets[0]
    elif dataset_number < 1:
        raise ValueError("concat_language_pair_dataset needs at least on dataset")
    # for dataset in language_pair_datasets:
    #     assert isinstance(dataset, LanguagePairDataset), "concat_language_pair_dataset can only concat language pair" \
    #                                                      "dataset"
    
    src_list = [language_pair_datasets[0].src]
    tgt_list = [language_pair_datasets[0].tgt]
    src_dict = language_pair_datasets[0].src_dict
    tgt_dict = language_pair_datasets[0].tgt_dict
    left_pad_source = language_pair_datasets[0].left_pad_source
    left_pad_target = language_pair_datasets[0].left_pad_target
    
    logger.info("To construct the source dataset list and the target dataset list")
    for dataset in language_pair_datasets[1:]:
        assert dataset.src_dict == src_dict
        assert dataset.tgt_dict == tgt_dict
        assert dataset.left_pad_source == left_pad_source
        assert dataset.left_pad_target == left_pad_target
        src_list.append(dataset.src)
        tgt_list.append(dataset.tgt)
    logger.info("Have constructed the source dataset list and the target dataset list")
    
    if all_dataset_upsample_ratio is None:
        sample_ratio = [1] * len(src_list)
        sample_ratio[0] = up_sample_ratio
    else:
        sample_ratio = [int(t) for t in all_dataset_upsample_ratio.strip().split(",")]
        assert len(sample_ratio) == len(src_list)
    src_dataset = ConcatDataset(src_list, sample_ratios=sample_ratio)
    tgt_dataset = ConcatDataset(tgt_list, sample_ratios=sample_ratio)
    res = LanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
    )
    logger.info("Have created the concat language pair dataset")
    return res


@register_task('translation_w_mono')
class TranslationWithMonoTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """
    
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--mono-data', default=None, help='monolingual data, split by :')
        parser.add_argument('--mono-one-split-each-epoch', action='store_true', default=False, help='use on split of monolingual data at each epoch')
        parser.add_argument('--parallel-ratio', default=1.0, type=float, help='subsample ratio of parallel data')
        parser.add_argument('--mono-ratio', default=1.0, type=float, help='subsample ratio of mono data')
        parser.add_argument('--do_shuf', action='store_true', default=False, help='shuffle data before iteration')
        parser.add_argument('--only_mono', action='store_true', default=False, help='Only monolingual data to be used for training')
        parser.add_argument('--only_parallel', action='store_true', default=False, help='Only parallel data to be used for training')
    
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.do_shuf = args.do_shuf
        self.update_number = 0
    
    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'
        
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')
        
        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        
        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        logger.info("To load the dataset {}".format(split))
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != getattr(self.cfg, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]
        print (f"only_parallel, {self.cfg.only_parallel} only_mono, {self.cfg.only_mono}")
        if not self.cfg.only_parallel:
            mono_paths = utils.split_paths(self.cfg.mono_data)
        
        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        
        if not self.cfg.only_mono:
            parallel_data = load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=combine, dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=self.cfg.left_pad_source,
                left_pad_target=self.cfg.left_pad_target,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=False,
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )
        if split == "train" and not self.cfg.only_parallel:
            if not self.cfg.only_mono:
                parallel_data = SubsampleLanguagePairDataset(parallel_data, size_ratio=self.cfg.parallel_ratio,
                                                         seed=self.cfg.seed,
                                                         epoch=epoch)
            if self.cfg.mono_one_split_each_epoch:
                mono_path = mono_paths[(epoch - 1) % len(mono_paths)]  # each at one epoch
                mono_data = load_langpair_dataset(
                    mono_path, split, src, self.src_dict, tgt, self.tgt_dict,
                    combine=combine, dataset_impl=self.cfg.dataset_impl,
                    upsample_primary=self.cfg.upsample_primary,
                    left_pad_source=self.cfg.left_pad_source,
                    left_pad_target=self.cfg.left_pad_target,
                    max_source_positions=self.cfg.max_source_positions,
                    shuffle=False,
                    max_target_positions=self.cfg.max_target_positions,
                )
                mono_data = SubsampleLanguagePairDataset(mono_data, size_ratio=self.cfg.mono_ratio,
                                                         seed=self.cfg.seed,
                                                         epoch=epoch)
                if self.cfg.only_mono:
                    all_dataset = [mono_data]
                else:
                    all_dataset = [parallel_data, mono_data]
            else:
                mono_datas = []
                for mono_path in mono_paths:
                    mono_data = load_langpair_dataset(
                        mono_path, split, src, self.src_dict, tgt, self.tgt_dict,
                        combine=combine, dataset_impl=self.cfg.dataset_impl,
                        upsample_primary=self.cfg.upsample_primary,
                        left_pad_source=self.cfg.left_pad_source,
                        left_pad_target=self.cfg.left_pad_target,
                        max_source_positions=self.cfg.max_source_positions,
                        shuffle=False,
                        max_target_positions=self.cfg.max_target_positions,
                    )
                    mono_data = SubsampleLanguagePairDataset(mono_data, size_ratio=self.cfg.mono_ratio,
                                                             seed=self.cfg.seed,
                                                             epoch=epoch)
                    mono_datas.append(mono_data)
                if self.cfg.only_mono:
                    all_dataset = mono_datas
                    self.datasets[split] = all_dataset
                else:
                    all_dataset =  mono_datas + [parallel_data]
                    self.datasets[split] = ConcatDataset(all_dataset)
        else:
            self.datasets[split] = parallel_data
    
    def strip_bos(self, samples):
        def _is_lang_id(idx):
            return idx > 32750
        # print ("Target", [self.src_dict.string(el) for el in samples[0]["target"][:40]])
        # print ("Prev output tokens", [self.src_dict.string(el) for el in samples[0]["net_input"]["prev_output_tokens"][:40]])
        # print ("Source tokens", [self.src_dict.string(el) for el in samples[0]["net_input"]["src_tokens"][:40]])
        # print ("majju",self.dicts["en"].string(torch.tensor([2])))
        # samples[0]["net_input"]["prev_output_tokens"] = samples[0]["net_input"]["prev_output_tokens"][:,1:]
        # samples[0]["target"] = samples[0]["target"][:,1:]
        # print ("After", samples)
        return
    
    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        print ("Shuffle: ", self.do_shuf)
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
            disable_shuffling=not self.do_shuf
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter