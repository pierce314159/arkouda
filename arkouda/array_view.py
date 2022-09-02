from __future__ import annotations

import json
from enum import Enum

import numpy as np  # type: ignore

from arkouda.client import generic_msg
from arkouda.dtypes import resolve_scalar_dtype, translate_np_dtype
from arkouda.numeric import cast as akcast
from arkouda.numeric import cumprod, where
from arkouda.pdarrayclass import create_pdarray, parse_single_value, pdarray
from arkouda.pdarraycreation import arange, array, ones, zeros
from arkouda.pdarrayIO import read_hdf5_multi_dim, write_hdf5_multi_dim
from arkouda.pdarraysetops import concatenate

OrderType = Enum("OrderType", ["ROW_MAJOR", "COLUMN_MAJOR"])


class ArrayView:
    """
    A multi-dimensional view of a pdarray. Arkouda ``ArraryView`` behaves similarly to numpy's ndarray.
    The base pdarray is stored in 1-dimension but can be indexed and treated logically
    as if it were multi-dimensional

    Attributes
    ----------
    base: pdarray
        The base pdarray that is being viewed as a multi-dimensional object
    dtype: dtype
        The element type of the base pdarray (equivalent to base.dtype)
    size: int_scalars
        The number of elements in the base pdarray (equivalent to base.size)
    shape: pdarray[int]
        A pdarray specifying the sizes of each dimension of the array
    ndim: int_scalars
         Number of dimensions (equivalent to shape.size)
    itemsize: int_scalars
        The size in bytes of each element (equivalent to base.itemsize)
    order: str {'C'/'row_major' | 'F'/'column_major'}
        Index order to read and write the elements.
        By default or if 'C'/'row_major', read and write data in row_major order
        If 'F'/'column_major', read and write data in column_major order
    """

    def __init__(self, base: pdarray, shape, order="row_major"):
        self.objtype = type(self).__name__
        self.shape = array(shape)
        if not isinstance(self.shape, pdarray):
            raise TypeError(f"ArrayView Shape cannot be type {type(self.shape)}. Expecting pdarray.")
        if base.size != self.shape.prod():
            raise ValueError(f"cannot reshape array of size {base.size} into shape {self.shape}")
        self.base = base
        self.size = base.size
        self.dtype = base.dtype
        self.ndim = self.shape.size
        self.itemsize = self.base.itemsize
        if order.upper() in {"C", "ROW_MAJOR"}:
            self.order = OrderType.ROW_MAJOR
        elif order.upper() in {"F", "COLUMN_MAJOR"}:
            self.order = OrderType.COLUMN_MAJOR
        else:
            raise ValueError(f"cannot traverse with order={order}")
        # cache _reverse_shape which is reversed if we're row_major
        self._reverse_shape = self.shape if self.order is OrderType.COLUMN_MAJOR else self.shape[::-1]
        if self.shape.min() == 0:
            # avoid divide by 0 if any of the dimensions are 0
            self._dim_prod = zeros(self.shape.size, self.dtype)
        else:
            # cache dim_prod to avoid recalculation, reverse if row_major
            self._dim_prod = (
                cumprod(self.shape) // self.shape
                if self.order is OrderType.COLUMN_MAJOR
                else cumprod(self._reverse_shape) // self._reverse_shape
            )

    def __len__(self):
        return self.size

    def __repr__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            return self.to_ndarray().__repr__()
        else:
            edge_items = np.get_printoptions()["edgeitems"]
            vals = [f"'{self.base[i]}'" for i in range(edge_items)]
            vals.append("... ")
            vals.extend([f"'{self.base[i]}'" for i in range(self.size - edge_items, self.size)])
        return f"array([{', '.join(vals)}]), shape {self.shape}"

    def __str__(self):
        from arkouda.client import pdarrayIterThresh

        if self.size <= pdarrayIterThresh:
            return self.to_ndarray().__str__()
        else:
            edge_items = np.get_printoptions()["edgeitems"]
            vals = [f"'{self.base[i]}'" for i in range(edge_items)]
            vals.append("... ")
            vals.extend([f"'{self.base[i]}'" for i in range(self.size - edge_items, self.size)])
        return f"[{', '.join(vals)}], shape {self.shape}"

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)
        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
        if len(key) < self.ndim:
            # append self.ndim-len(key) many ':'s to fill in the missing dimensions
            for i in range(self.ndim - len(key)):
                key.append(slice(None, None, None))
        try:
            # attempt to convert to a pdarray (allows for view[0,2,1] instead of view[ak.array([0,2,1])]
            # but pass on RuntimeError to allow things like
            # view[0,:,[True,False,True]] to be correctly handled
            if not any([isinstance(i, (list, slice)) for i in key]):
                key = array(key)
        except (RuntimeError, TypeError, ValueError, DeprecationWarning):
            pass
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("int", "uint", "bool"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool":
                if key.all():
                    # every dimension is True, so return this arrayview with shape = [1, self.shape]
                    return self.base.reshape(
                        concatenate([ones(1, dtype=self.dtype), self.shape]), order=self.order.name
                    )
                else:
                    # at least one dimension is False,
                    # so return empty arrayview with shape = [0, self.shape]
                    return array([], dtype=self.dtype).reshape(
                        concatenate([zeros(1, dtype=self.dtype), self.shape]), order=self.order.name
                    )
            # Interpret negative key as offset from end of array
            key = where(key < 0, akcast(key + self.shape, kind), key)
            # Capture the indices which are still out of bounds
            out_of_bounds = (key < 0) | (self.shape <= key)
            if out_of_bounds.any():
                out = arange(key.size)[out_of_bounds][0]
                raise IndexError(
                    f"index {key[out]} is out of bounds for axis {out} with size {self.shape[out]}"
                )
            coords = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
            repMsg = generic_msg(
                cmd="arrayViewIntIndex",
                args={
                    "base": self.base,
                    "dim_prod": self._dim_prod,
                    "coords": coords,
                },
            )
            fields = repMsg.split()
            return parse_single_value(" ".join(fields[1:]))
        elif isinstance(key, list):
            types = []
            coords = []
            reshape_dim_list = []
            index_dim_list = []  # maybe called coordinate_dims instead of index/user dims
            advanced = []
            reshape_advanced = []
            arrays = []  # kinda hate this but we don't want unknown sym errors
            key = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
            for i, x in enumerate(key):
                if np.isscalar(x) and (resolve_scalar_dtype(x) in ["int64", "uint64"]):
                    orig_key = x
                    if x < 0:
                        # Interpret negative key as offset from end of array
                        x += self._reverse_shape[i]
                    if 0 <= x < self._reverse_shape[i]:
                        types.append("int")
                        # have to cast to int because JSON doesn't recognize numpy dtypes
                        coords.append(json.dumps(int(x)))
                        index_dim_list.append(1)
                        advanced.append(False)
                    else:
                        raise IndexError(
                            f"index {orig_key} is out of bounds for axis {i} "
                            f"with size {self._reverse_shape[i]}"
                        )
                elif isinstance(x, slice):
                    (start, stop, stride) = x.indices(self._reverse_shape[i])
                    types.append("slice")
                    coords.append(json.dumps((start, stop, stride)))
                    slice_size = len(range(*(start, stop, stride)))
                    index_dim_list.append(slice_size)
                    reshape_dim_list.append(slice_size)
                    advanced.append(False)
                    reshape_advanced.append(False)
                elif isinstance(x, pdarray) or isinstance(x, list):
                    # raise TypeError(f"Advanced indexing is not yet supported {x} ({type(x)})")

                    # UGGGHGHHHH i prob gotta do the same preprocessing as the pdarray case
                    # if bool
                    # if there are integer arrays present:
                    #     treat as integer special array (i.e. indices where non-zero):
                    #     [i for i,k in enumerate(b2) if k!=0]
                    # elif has the same shape as self:
                    #     select only indices where true in flattened array
                    # else if shape differs:
                    #     act as if we are [b, ...], i.e. index by b followed by as many ':' as are needed
                    #     to fill out the rank of x

                    x = array(x)
                    kind, _ = translate_np_dtype(x.dtype)
                    if kind not in ("bool", "int"):
                        raise TypeError("unsupported pdarray index type {}".format(x.dtype))
                    # if kind == "bool" and dim != x.size:
                    #     raise ValueError("size mismatch {} {}".format(dim, x.size))
                    types.append("pdarray")
                    coords.append(x.name)
                    index_dim_list.append(x.size)
                    reshape_dim_list.append(x.size)
                    advanced.append(True)
                    reshape_advanced.append(True)
                    arrays.append(x)
                else:
                    raise TypeError(f"Unhandled key type: {x} ({type(x)})")

            advanced = (
                np.array(advanced[::-1]) if self.order is OrderType.COLUMN_MAJOR else np.array(advanced)
            )
            reshape_advanced = (
                np.array(reshape_advanced[::-1]) if self.order is OrderType.COLUMN_MAJOR else np.array(reshape_advanced)
            )
            is_non_consecutive = ((advanced[0] and advanced[-1]) and not all(advanced)) or sum(
                np.logical_xor(advanced, list(advanced[1:]) + [advanced[-1]])
            ) > 2

            reshape_dim = ~reshape_advanced
            first_advanced = np.argmax(reshape_advanced)
            reshape_dim[first_advanced] = True
            reshape_dim = reshape_dim[::-1] if self.order is OrderType.COLUMN_MAJOR else reshape_dim
            intermediary_user_dims = np.where(reshape_dim, reshape_dim_list, 1)
            advanced_len = reshape_dim_list[first_advanced]
            if is_non_consecutive:
                # if non-consecutive special indicies
                # remove first special and add len special to front
                intermediary_user_dims = (
                    [advanced_len]
                    + list(intermediary_user_dims[:first_advanced])
                    + list(intermediary_user_dims[(first_advanced + 1) :])
                )
            user_dim_prod = array(list(np.cumprod(intermediary_user_dims) // intermediary_user_dims))

            reshape_dim_list = np.array(reshape_dim_list)
            if is_non_consecutive:
                reshape_dim_list = [reshape_dim_list[reshape_advanced][0]] + list(
                    reshape_dim_list[~reshape_advanced[::-1]]
                )
                reshape_dim_list = reshape_dim_list[::-1]
                print(f"reshape_dim_list = {reshape_dim_list}")
            else:
                reshape_dim_list = reshape_dim_list[reshape_dim]
            ret_size = np.prod(reshape_dim_list)
            reshape_dim = array(list(reshape_dim))
            advanced = array(list(advanced))
            print(f"user_dim_prod = {user_dim_prod}")
            print(f"reshape_dim = {reshape_dim}")
            print(f"advanced = {advanced}")
            print(f"reshape_advanced = {reshape_advanced}")

            index_dim = array(list(index_dim_list))
            repMsg = generic_msg(
                cmd="arrayViewMixedIndex",
                args={
                    "base": self.base,
                    "index_dim": index_dim,
                    "ndim": self.ndim,
                    "dim_prod": self._dim_prod,
                    "types": types,
                    "coords": coords,
                    "user_dim_prod": user_dim_prod,
                    "reshape_dim": reshape_dim,
                    "advanced": advanced,
                    "advanced_len": advanced_len,
                    "is_non_consecutive": is_non_consecutive,
                    "ret_size": ret_size,
                },
            )

            # IVE NO CLUE WHAT TO DO NOW WHEN NOT COLUMN MAJOR
            reshape_dim = (
                reshape_dim_list if self.order is OrderType.COLUMN_MAJOR else reshape_dim_list[::-1]
            )
            print(f"reshape_dim_list = {reshape_dim_list}")
            print(f"ORDER = {self.order}")
            print(f"reshape_dim = {reshape_dim}")

            # see if this fixes the unknown sym stuff
            inter_arr = create_pdarray(repMsg)
            return inter_arr.reshape(list(reshape_dim), order=self.order.name)
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def __setitem__(self, key, value):
        if isinstance(key, int) or isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)
        if len(key) > self.ndim:
            raise IndexError(
                f"too many indices for array: array is {self.ndim}-dimensional, "
                f"but {len(key)} were indexed"
            )
        if len(key) < self.ndim:
            # append self.ndim-len(key) many ':'s to fill in the missing dimensions
            for i in range(self.ndim - len(key)):
                key.append(slice(None, None, None))
        try:
            # attempt to convert to a pdarray (allows for view[0,2,1] instead of view[ak.array([0,2,1])]
            # but pass on RuntimeError to allow things like
            # view[0,:,[True,False,True]] to be correctly handled
            key = array(key)
        except (RuntimeError, TypeError, ValueError, DeprecationWarning):
            pass
        if isinstance(key, pdarray):
            kind, _ = translate_np_dtype(key.dtype)
            if kind not in ("int", "uint", "bool"):
                raise TypeError(f"unsupported pdarray index type {key.dtype}")
            if kind == "bool":
                if key.all():
                    # every dimension is True, so fill arrayview with value
                    # if any dimension is False, we don't update anything
                    self.base.fill(value)
            else:
                # Interpret negative key as offset from end of array
                key = where(key < 0, akcast(key + self.shape, kind), key)
                # Capture the indices which are still out of bounds
                out_of_bounds = (key < 0) | (self.shape <= key)
                if out_of_bounds.any():
                    out = arange(key.size)[out_of_bounds][0]
                    raise IndexError(
                        f"index {key[out]} is out of bounds for axis {out} with size {self.shape[out]}"
                    )
                coords = key if self.order is OrderType.COLUMN_MAJOR else key[::-1]
                generic_msg(
                    cmd="arrayViewIntIndexAssign",
                    args={
                        "base": self.base,
                        "dtype": self.dtype,
                        "dim_prod": self._dim_prod,
                        "coords": coords,
                        "value": self.base.format_other(value),
                    },
                )
        elif isinstance(key, list):
            raise NotImplementedError("Setting via slicing and advanced indexing is not yet supported")
        else:
            raise TypeError(f"Unhandled key type: {key} ({type(key)})")

    def to_ndarray(self) -> np.ndarray:
        """
        Convert the ArrayView to a np.ndarray, transferring array data from the
        Arkouda server to client-side Python. Note: if the ArrayView size exceeds
        client.maxTransferBytes, a RuntimeError is raised.

        Returns
        -------
        np.ndarray
            A numpy ndarray with the same attributes and data as the ArrayView

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown, if the ArrayView size
            exceeds the built-in client.maxTransferBytes size limit, or if the bytes
            received does not match expected number of bytes
        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        array()
        to_list()

        Examples
        --------
        >>> a = ak.arange(6).reshape(2,3)
        >>> a.to_ndarray()
        array([[0, 1, 2],
               [3, 4, 5]])
        >>> type(a.to_ndarray())
        numpy.ndarray
        """
        if self.order is OrderType.ROW_MAJOR:
            return self.base.to_ndarray().reshape(self.shape.to_ndarray())
        else:
            return self.base.to_ndarray().reshape(self.shape.to_ndarray(), order="F")

    def to_list(self) -> list:
        """
        Convert the ArrayView to a list, transferring array data from the
        Arkouda server to client-side Python. Note: if the ArrayView size exceeds
        client.maxTransferBytes, a RuntimeError is raised.

        Returns
        -------
        list
            A list with the same data as the ArrayView

        Raises
        ------
        RuntimeError
            Raised if there is a server-side error thrown, if the ArrayView size
            exceeds the built-in client.maxTransferBytes size limit, or if the bytes
            received does not match expected number of bytes

        Notes
        -----
        The number of bytes in the array cannot exceed ``client.maxTransferBytes``,
        otherwise a ``RuntimeError`` will be raised. This is to protect the user
        from overflowing the memory of the system on which the Python client
        is running, under the assumption that the server is running on a
        distributed system with much more memory than the client. The user
        may override this limit by setting client.maxTransferBytes to a larger
        value, but proceed with caution.

        See Also
        --------
        to_ndarray()

        Examples
        --------
        >>> a = ak.arange(6).reshape(2,3)
        >>> a.to_list()
        [[0, 1, 2], [3, 4, 5]]
        >>> type(a.to_list())
        list
        """
        return self.to_ndarray().tolist()

    def save(self, filepath: str, dset: str, mode: str = "truncate", storage: str = "Flat"):
        """
        Save the current ArrayView object to hdf5 file

        Parameters
        ----------
        filepath: str
            Path to the file to write the dataset to
        dset: str
            Name of the dataset to write
        mode: str (truncate | append)
            Default: truncate
            Mode to write the dataset in. Truncate will overwrite any existing files.
            Append will add the dataset to an existing file.
        storage: str (Flat | Multi)
            Default: Flat
            Method to use when storing the dataset.
            Flat - flatten the multi-dimensional object into a 1-D array of values
            Multi - Store the object in the multidimensional presentation.

        See Also
        --------
        ak.ArrayView.load
        """
        write_hdf5_multi_dim(self, filepath, dset, mode=mode, storage=storage)

    @staticmethod
    def load(filepath: str, dset: str) -> ArrayView:
        """
        Read a multi-dimensional dataset from an HDF5 file into an ArrayView object

        Parameters
        ----------
        file_path: str
            path to the file to read from
        dset: str
            name of the dataset to read

        Returns
        -------
        ArrayView object representing the data read from file

        See Also
        --------
        ak.ArrayView.save
        """
        return read_hdf5_multi_dim(filepath, dset)
