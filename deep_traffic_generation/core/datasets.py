# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from traffic.core import Traffic

from multiprocessing import Pool
import itertools
import os

from .protocols import TransformerProtocol

from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


# fmt: on
class Infos(TypedDict):
    features: List[str]
    index: Optional[int]


class DatasetParams(TypedDict):
    features: List[str]
    file_path: Optional[Path]
    info_params: Infos
    input_dim: int
    scaler: Optional[TransformerProtocol]
    seq_len: int
    shape: str


# fmt: on
class TrafficDataset(Dataset):
    """Traffic Dataset

    Args:
        traffic: Traffic object to extract data from.
        features: features to extract from traffic.
        shape (optional): shape of datapoints when:

            - ``'image'``: tensor of shape
              :math:`(\\text{feature}, \\text{seq})`.
            - ``'linear'``: tensor of shape
              :math:`(\\text{feature} \\times \\text{seq})`.
            - ``'sequence'``: tensor of shape
              :math:`(\\text{seq}, \\text{feature})`. Defaults to
              ``'sequence'``.
        scaler (optional): scaler to apply to the data. You may want to
            consider `StandardScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
            Defaults to None.
        info_params (Infos, optional): typed dictionnary with two keys:
            `features` (List[str]): list of features.
            `index` (int): index in the underlying trajectory DataFrame
            where to get the features.
            Defaults ``features=[]`` and ``index=None``.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]

    def __init__(
        self,
        traffic: Traffic,
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> None:

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. "
            + f"Available shapes are: {self._available_shapes}"
        )

        self.file_path: Optional[Path] = None
        self.features = features
        self.shape = shape
        self.scaler = scaler
        self.info_params = info_params

        self.data: torch.Tensor
        self.lengths: List[int]
        self.infos: List[Any]
        # self.target_transform = target_transform

        # extract features
        # data = extract_features(traffic, features, info_params["features"])
        data = np.stack(
            list(np.append(f.flight_id, f.data[self.features].values.ravel()) for f in traffic)
        )
        
        #keep_track of flight_id
        self.flight_ids = list(data[:,0])
        data = data[:,1:].astype(float)

        self.scaler = scaler
        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                data = self.scaler.transform(data)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)

        data = torch.FloatTensor(data)

        self.data = data
        if self.shape in ["sequence", "image"]:
            self.data = self.data.view(
                self.data.size(0), -1, len(self.features)
            )
            if self.shape == "image":
                self.data = torch.transpose(self.data, 1, 2)

        # gives info nedeed to reconstruct the trajectory
        # info_params["index"] = -1 means we need the coordinates of the last position
        self.infos = []
        # TODO: change condition (if not is_empty(self.info_params))
        if self.info_params["index"] is not None:
            self.infos = torch.Tensor(
                [
                    f.data[self.info_params["features"]]
                    .iloc[self.info_params["index"]]
                    .values.ravel()
                    for f in traffic
                ]
            )

    @classmethod
    def from_file(
        cls,
        file_path: Union[str, Path],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> "TrafficDataset":
        file_path = (
            file_path if isinstance(file_path, Path) else Path(file_path)
        )
        traffic = Traffic.from_file(file_path)
        dataset = cls(traffic, features, shape, scaler, info_params)
        dataset.file_path = file_path
        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, List[Any]]:
        """Get item method, returns datapoint at some index.

        Args:
            index (int): An index. Should be :math:`<len(self)`.

        Returns:
            torch.Tensor: The trajectory data shaped accordingly to self.shape.
            int: The length of the trajectory.
            list: List of informations that could be needed like, labels or
                original latitude and longitude values.
        """
        infos = []
        if self.info_params["index"] is not None:
            infos = self.infos[index]
        return self.data[index], infos
    
    def get_flight(self, flight_id: str) -> torch.Tensor:
        index = self.flight_ids.index(flight_id)
        return self.data[index]

    @property
    def input_dim(self) -> int:
        """Returns the size of datapoint's features.

        .. warning::
            If the `self.shape` is ``'linear'``, the returned size will be
            :math:`\\text{feature_n} \\times \\text{sequence_len}`
            since the temporal dimension is not taken into account with this
            shape.
        """
        if self.shape in ["linear", "sequence"]:
            return self.data.shape[-1]
        elif self.shape == "image":
            return self.data.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data.shape[1]
        elif self.shape == "image":
            return self.data.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * info_params (TypedDict) (see Infos for details)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            features=self.features,
            file_path=self.file_path,
            info_params=self.info_params,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            shape=self.shape,
        )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds TrafficDataset arguments to ArgumentParser.

        List of arguments:

            * ``--data_path``: The path to the traffic data file. Default to
              None.
            * ``--features``: The features to keep for training. Default to
              ``['latitude','longitude','altitude','timedelta']``.

              Usage:

              .. code-block:: console

                python main.py --features track groundspeed altitude timedelta

            * ``--info_features``: Features not passed through the model but
              useful to keep. For example, if you chose as main features:
              track, groundspeed, altitude and timedelta ; it might help to
              keep the latitude and longitude values of the first or last
              coordinates to reconstruct the trajectory. The values are picked
              up at the index specified at ``--info_index``. You can also
              get some labels.

              Usage:

              .. code-block:: console

                python main.py --info_features latitude longitude

                python main.py --info_features label

            * ``--info_index``: Index of information features. Default to None.

        Args:
            parser (ArgumentParser): ArgumentParser to update.

        Returns:
            ArgumentParser: updated ArgumentParser with TrafficDataset
            arguments.
        """
        p = parser.add_argument_group("TrafficDataset")
        p.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            default=None,
        )
        p.add_argument(
            "--features",
            dest="features",
            nargs="+",
            default=["latitude", "longitude", "altitude", "timedelta"],
        )
        p.add_argument(
            "--info_features",
            dest="info_features",
            nargs="+",
            default=[],
        )
        p.add_argument(
            "--info_index",
            dest="info_index",
            type=int,
            default=None,
        )
        return parser

#Method to calculate pairs of trajectories
def calculate_pairs(iter):
    x, y = iter
    delta_t = y[1] - x[1]
    len_x = x[2] - x[1]
    len_y = y[2] - y[1]

    #modified according to sandobox_tcas
    if delta_t < -len_y or (delta_t > len_x):
        return
    
    #make sure that delta_t is smaller than the total duration of the reference (the takeoff)
    elif (delta_t < len_x): 
        return np.concatenate(([x[0], y[0], delta_t.total_seconds()], x[3:], y[3:]))

# fmt: on
class TrafficDatasetPairs(Dataset):
    """Traffic Dataset for pairs of trajectories
    delta_t: time difference between each trajectory within a pair is calculated
    Datapoints : list(delta_t, tensor)

    Args:
        traffic1: Traffic object to extract data from.
        traffic2: Traffic object to extract data from.
        features: features to extract from traffic.
        shape (optional): shape of trajectories pairs when:

            - ``'image'``: tensor of shape
              :math:`(2 \\times \\text{feature}, \\text{seq})`.
            - ``'linear'``: tensor of shape
              :math:`(2 \\times \\text{feature} \\times \\text{seq} + 1)`.
            - ``'sequence'``: tensor of shape
              :math:`(\\text{seq}, 2 \\times \\text{feature})`. Defaults to
              ``'sequence'``.
        scaler (optional): scaler to apply to the data. You may want to
            consider `StandardScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
            Defaults to None.
        info_params (Infos, optional): typed dictionnary with two keys:
            `features` (List[str]): list of features.
            `index` (int): index in the underlying trajectory DataFrame
            where to get the features.
            Defaults ``features=[]`` and ``index=None``.
    """

    _repr_indent = 4
    _available_shapes = ["linear", "sequence", "image"]

    def __init__(
        self,
        traffic1: Traffic,
        traffic2: Traffic,
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> None:

        assert shape in self._available_shapes, (
            f"{shape} shape is not available. "
            + f"Available shapes are: {self._available_shapes}"
        )

        self.file_path: Optional[Path] = None
        self.features = features
        self.shape = shape
        self.scaler = scaler
        self.info_params = info_params

        self.data: torch.Tensor
        self.lengths: List[int]
        self.infos: List[Any]

        # extract features for each traffic object
        data1 = np.stack(
            list(np.append([f.flight_id, f.start, f.stop], f.data[self.features].values.ravel()) for f in traffic1)
        )
        data2 = np.stack(
            list(np.append([f.flight_id, f.start, f.stop], f.data[self.features].values.ravel()) for f in traffic2)
        )

        #Forms pairs of trajectories and calculate delta_t

        with Pool(processes=os.cpu_count()) as p: 
            pairs = p.map(calculate_pairs, itertools.product(data1,data2))
            p.close()
            p.join()

        data = np.stack([x for x in pairs if x is not None])
        self.pairs_id  = data[:,:2]
        # data = data[:,2:].astype(float) # with delta t
        data = data[:,3:].astype(float) # without delta_t

        self.scaler = scaler
        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                data = self.scaler.transform(data)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)

        data = torch.FloatTensor(data)

        # self.data_pairs = data[:,1:]
        # self.delta_t = data[:,0]
        self.data_pairs = data #without delta_t
        if self.shape in ["sequence", "image"]:
            self.data1 = self.data_pairs[:,:int(self.data_pairs.size(1)/2)].view(
                self.data_pairs.size(0), -1, len(self.features)
            )
            self.data2 = self.data_pairs[:,int(self.data_pairs.size(1)/2):].view(
                self.data_pairs.size(0), -1, len(self.features)
            )
            # self.data_pairs = torch.cat((first,second), dim = 2)
            if self.shape == "image":
                self.data1 = torch.transpose(self.data1, 1, 2)
                self.data2 = torch.transpose(self.data2, 1, 2)

    @classmethod
    def from_file(
        cls,
        file_path: Tuple[Union[str, Path], Union[str, Path]],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> "TrafficDataset":
        file_path1 = (
            file_path[0] if isinstance(file_path[0], Path) else Path(file_path[0])
        )
        file_path2 = (
            file_path[1] if isinstance(file_path[1], Path) else Path(file_path[1])
        )
        traffic1 = Traffic.from_file(file_path1)
        traffic2 = Traffic.from_file(file_path2)

        dataset = cls(traffic1, traffic2, features, shape, scaler, info_params)
        dataset.file_path = (file_path1, file_path2)
        return dataset

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item method, returns datapoint at some index.

        Args:
            index (int): An index. Should be :math:`<len(self)`.

        Returns:
            torch.Tensor: First trajectory data shaped accordingly to self.shape.
            torch.Tensor: Second trajectory data shaped accordingly to self.shape.
            torch.Tensor: delta_t between the two trajectories of the pair
        """
        # return self.data_pairs[index], self.delta_t[index]
        return self.data1[index], self.data2[index]#, self.delta_t[index]

    @property
    def input_dim(self) -> int:
        """Returns the size of datapoint's features.

        .. warning::
            If the `self.shape` is ``'linear'``, the returned size will be
            :math:`\\text{feature_n} \\times \\text{sequence_len}`
            since the temporal dimension is not taken into account with this
            shape.
        """
        if self.shape == "linear":
            return self.data_pairs.shape[-1]
        elif self.shape in "sequence":
            return self.data1.shape[-1]
        elif self.shape == "image":
            return self.data1.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data1.shape[1]
        elif self.shape == "image":
            return self.data1.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * info_params (TypedDict) (see Infos for details)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            features=self.features,
            files_paths=self.file_path,
            info_params=self.info_params,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            shape=self.shape,
        )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds TrafficDataset arguments to ArgumentParser.

        List of arguments:

            * ``--data_path``: The path to the traffic data file. Default to
              None.
            * ``--features``: The features to keep for training. Default to
              ``['latitude','longitude','altitude','timedelta']``.

              Usage:

              .. code-block:: console

                python main.py --features track groundspeed altitude timedelta

            * ``--info_features``: Features not passed through the model but
              useful to keep. For example, if you chose as main features:
              track, groundspeed, altitude and timedelta ; it might help to
              keep the latitude and longitude values of the first or last
              coordinates to reconstruct the trajectory. The values are picked
              up at the index specified at ``--info_index``. You can also
              get some labels.

              Usage:

              .. code-block:: console

                python main.py --info_features latitude longitude

                python main.py --info_features label

            * ``--info_index``: Index of information features. Default to None.

        Args:
            parser (ArgumentParser): ArgumentParser to update.

        Returns:
            ArgumentParser: updated ArgumentParser with TrafficDataset
            arguments.
        """
        p = parser.add_argument_group("TrafficDatasetPairs")
        p.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            nargs=2,
            default=None,
        )
        p.add_argument(
            "--features",
            dest="features",
            nargs="+",
            default=["latitude", "longitude", "altitude", "timedelta"],
        )
        p.add_argument(
            "--info_features",
            dest="info_features",
            nargs="+",
            default=[],
        )
        p.add_argument(
            "--info_index",
            dest="info_index",
            type=int,
            default=None,
        )
        return parser
    

class MetaDatasetPairs(Dataset):    
    
    """Dataset for pairs of latent space representations of trajectories for MetaVAE  
    Datapoints : list(tensor, tensor)

    Args:
        trafficDataset1: TrafficDataset object for VAE1.
        trafficDataset2: TrafficDataset object for VAE2.
        vae1_path: path to VAE trained to represent generate trajectories from traffic1.
        vae2_path: path to VAE trained to represent generate trajectories from traffic2.
        scaler (optional): scaler to apply to the data. You may want to
            consider `StandardScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_.
            Defaults to None.
    """

    _repr_indent = 4

    def __init__(
        self,
        trafficDataset1: TrafficDataset,
        trafficDataset2: TrafficDataset,
        vae1_path: str,
        vae2_path: str, 
        scaler: Optional[TransformerProtocol] = None,
    ) -> None:
        
        self.trafficDataset1 = trafficDataset1
        self.trafficDataset2 = trafficDataset2
        
        data1 = trafficDataset1.scaler.inverse_trasform(trafficDataset1.data)
        data2 = trafficDataset2.scaler.inverse_trasform(trafficDataset2.data)

        self.VAE1 = TCVAE.load_from_checkpoint(
            vae1_path + "checkpoints/" + filenames[0],
            dataset_params=trafficDataset1.parameters,
        )
        
        self.VAE2 = TCVAE.load_from_checkpoint(
            vae2_path + "checkpoints/" + filenames[0],
            dataset_params=trafficDataset2.parameters,
        )

        #Forms pairs of trajectories and calculate delta_t

        with Pool(processes=os.cpu_count()) as p: 
            pairs = p.map(calculate_pairs, itertools.product(data1,data2))
            p.close()
            p.join()

        data = np.stack([x for x in pairs if x is not None])
        self.pairs_id  = data[:,:2]
        # data = data[:,2:].astype(float) # with delta t
        data = data[:,3:].astype(float) # without delta_t

        self.scaler = scaler
        if self.scaler is not None:
            try:
                # If scaler already fitted, only transform
                check_is_fitted(self.scaler)
                data = self.scaler.transform(data)
            except NotFittedError:
                # If not: fit and transform
                self.scaler = self.scaler.fit(data)
                data = self.scaler.transform(data)

        data = torch.FloatTensor(data)

        # self.data_pairs = data[:,1:]
        # self.delta_t = data[:,0]
        self.data_pairs = data #without delta_t
        if self.shape in ["sequence", "image"]:
            self.data1 = self.data_pairs[:,:int(self.data_pairs.size(1)/2)].view(
                self.data_pairs.size(0), -1, len(self.features)
            )
            self.data2 = self.data_pairs[:,int(self.data_pairs.size(1)/2):].view(
                self.data_pairs.size(0), -1, len(self.features)
            )
            # self.data_pairs = torch.cat((first,second), dim = 2)
            if self.shape == "image":
                self.data1 = torch.transpose(self.data1, 1, 2)
                self.data2 = torch.transpose(self.data2, 1, 2)

    @classmethod
    def from_file(
        cls,
        file_path: Tuple[Union[str, Path], Union[str, Path]],
        features: List[str],
        shape: str = "linear",
        scaler: Optional[TransformerProtocol] = None,
        info_params: Infos = Infos(features=[], index=None),
    ) -> "TrafficDataset":
        file_path1 = (
            file_path[0] if isinstance(file_path[0], Path) else Path(file_path[0])
        )
        file_path2 = (
            file_path[1] if isinstance(file_path[1], Path) else Path(file_path[1])
        )
        traffic1 = Traffic.from_file(file_path1)
        traffic2 = Traffic.from_file(file_path2)

        dataset = cls(traffic1, traffic2, features, shape, scaler, info_params)
        dataset.file_path = (file_path1, file_path2)
        return dataset

    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item method, returns datapoint at some index.

        Args:
            index (int): An index. Should be :math:`<len(self)`.

        Returns:
            torch.Tensor: First trajectory data shaped accordingly to self.shape.
            torch.Tensor: Second trajectory data shaped accordingly to self.shape.
            torch.Tensor: delta_t between the two trajectories of the pair
        """
        # return self.data_pairs[index], self.delta_t[index]
        return self.data1[index], self.data2[index]#, self.delta_t[index]

    @property
    def input_dim(self) -> int:
        """Returns the size of datapoint's features.

        .. warning::
            If the `self.shape` is ``'linear'``, the returned size will be
            :math:`\\text{feature_n} \\times \\text{sequence_len}`
            since the temporal dimension is not taken into account with this
            shape.
        """
        if self.shape == "linear":
            return self.data_pairs.shape[-1]
        elif self.shape in "sequence":
            return self.data1.shape[-1]
        elif self.shape == "image":
            return self.data1.shape[1]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def seq_len(self) -> int:
        """Returns sequence length (i.e. maximum sequence length)."""
        if self.shape == "linear":
            return int(self.input_dim / len(self.features))
        elif self.shape == "sequence":
            return self.data1.shape[1]
        elif self.shape == "image":
            return self.data1.shape[2]
        else:
            raise ValueError(f"Invalid shape value: {self.shape}.")

    @property
    def parameters(self) -> DatasetParams:
        """Returns parameters of the TrafficDataset object in a TypedDict.

        * features (List[str])
        * file_path (Path, optional)
        * info_params (TypedDict) (see Infos for details)
        * input_dim (int)
        * scaler (Any object that matches TransformerProtocol, see TODO)
        * seq_len (int)
        * shape (str): either ``'image'``, ``'linear'`` or ```'sequence'``.
        """
        return DatasetParams(
            features=self.features,
            files_paths=self.file_path,
            info_params=self.info_params,
            input_dim=self.input_dim,
            scaler=self.scaler,
            seq_len=self.seq_len,
            shape=self.shape,
        )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        # if self.file_path is not None:
        #     body.append(f"File location: {self.file_path}")
        if self.scaler is not None:
            body += [repr(self.scaler)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser) -> ArgumentParser:
        """Adds TrafficDataset arguments to ArgumentParser.

        List of arguments:

            * ``--data_path``: The path to the traffic data file. Default to
              None.
            * ``--features``: The features to keep for training. Default to
              ``['latitude','longitude','altitude','timedelta']``.

              Usage:

              .. code-block:: console

                python main.py --features track groundspeed altitude timedelta

            * ``--info_features``: Features not passed through the model but
              useful to keep. For example, if you chose as main features:
              track, groundspeed, altitude and timedelta ; it might help to
              keep the latitude and longitude values of the first or last
              coordinates to reconstruct the trajectory. The values are picked
              up at the index specified at ``--info_index``. You can also
              get some labels.

              Usage:

              .. code-block:: console

                python main.py --info_features latitude longitude

                python main.py --info_features label

            * ``--info_index``: Index of information features. Default to None.

        Args:
            parser (ArgumentParser): ArgumentParser to update.

        Returns:
            ArgumentParser: updated ArgumentParser with TrafficDataset
            arguments.
        """
        p = parser.add_argument_group("TrafficDatasetPairs")
        p.add_argument(
            "--data_path",
            dest="data_path",
            type=Path,
            nargs=2,
            default=None,
        )
        p.add_argument(
            "--features",
            dest="features",
            nargs="+",
            default=["latitude", "longitude", "altitude", "timedelta"],
        )
        p.add_argument(
            "--info_features",
            dest="info_features",
            nargs="+",
            default=[],
        )
        p.add_argument(
            "--info_index",
            dest="info_index",
            type=int,
            default=None,
        )
        return parser
