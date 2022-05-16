
import json

import torch

from dataset import Dataset

from . import util

# TODO - inherit from custom dataset


class TilesDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, scoring_data_filepath, top_n=100):
        if not isinstance(underlying_dataset, Dataset):
            raise TypeError(type(underlying_dataset))
        self.underlying_dataset = underlying_dataset
        self.scoring_data_filepath = scoring_data_filepath
        self.top_n = top_n
        # load in scoring data
        self.scoring_data = None
        with util.open_file(self.scoring_data_filepath) as f:
            self.scoring_data = json.load(f)
        self._filepaths = list(self.scoring_data.keys())
        # preprocess from fields and verify that scoring data files are in dataset
        self._passing_region_counts = {}
        for filepath in self.scoring_data.keys():  # for each filepath in the tilesdataset
            if filepath not in self.underlying_dataset._region_counts:  # if filepath isn't in underlying dataset
                raise Exception(f"filepath not in dataset: {filepath}")
            self._passing_region_counts[filepath] = min(top_n, len(
                self.scoring_data[filepath]))  # otherwise record length
        self._len = sum(self._passing_region_counts.values())

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index, len(self))
        region_filename, region_index = None, None
        for filename, region_count in self._passing_region_counts.items():
            if index >= region_count:
                index -= region_count
            else:
                region_filename, region_index = filename, index
                break
        region_index = self.scoring_data[region_filename][region_index][1]
        region = self.underlying_dataset.get_region(
            region_filename, region_index)
        return region, self.underlying_dataset.get_label(region_filename)

    def get_region_labels_as_list(self):
        region_labels = []
        for filepath in self._filepaths:
            label = self.underlying_dataset.get_label(filepath)
            regions_in_filepath = self._passing_region_counts[filepath]
            region_labels.extend([label] * regions_in_filepath)
        return region_labels

    def get_label_distribution(self):
        label_distribution = {}
        for label in self.get_region_labels_as_list():
            if label not in label_distribution:
                label_distribution[label] = 0
            label_distribution[label] += 1
        return label_distribution

    def iterate_by_file(self, as_pytorch_datasets=False):
        if not as_pytorch_datasets:
            def regions_generator(filename: str):
                for i in range(self.number_of_regions(filename)):
                    yield self.underlying_dataset.get_region(filename, i)
            for filename in self._filepaths:
                yield filename, self.get_label(filename), regions_generator(filename)
        else:
            class SingleFileDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, filename) -> None:
                    self.base_dataset = base_dataset
                    self.filename = filename

                def __getitem__(self, index):
                    return self.base_dataset.underlying_dataset.get_region(self.filename, index)

                def __len__(self):
                    return self.base_dataset.number_of_regions(self.filename)
            for filename in self._filepaths:
                yield filename, self.get_label(filename), SingleFileDataset(base_dataset=self, filename=filename)

    def get_label(self, filename):
        return self.underlying_dataset.get_label(filename)

    def number_of_regions(self, filename=None):
        if filename is None:
            return len(self)
        else:
            return self._passing_region_counts[filename]
