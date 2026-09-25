import gzip
import pickle
import os
import re
from typing import Any, Dict, List, Tuple, Sequence
from collections import defaultdict, deque
import numpy as np
import pandas as pd

import preprocessing
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import HTML, display, clear_output
import folium
from folium.plugins import TimestampedGeoJson
import datetime
import warnings
from tqdm.notebook import tqdm

from learning_utils import process_over_branches, calculate_macro_averaged_accuracy


def save_macro_averaged_accuracy_table(
    predictions,
    label,
    task_type,
    number_of_attention_branches,
):

    def macro_averaged_accuracy_func(prediction_argmax, label_argmax):
        macro_averaged_accuracy = calculate_macro_averaged_accuracy(prediction_argmax, label_argmax)
        return macro_averaged_accuracy
    
    macro_averaged_accuracies_tensor, prediction_tensor_list, label_tensor_list, prediction_argmax_tensor_list, label_argmax_tensor_list = process_over_branches(
        predictions,
        label,
        task_type,
        number_of_attention_branches,
        func=macro_averaged_accuracy_func,
        use_argmax_for_func=True,
    )
    macro_averaged_accuracies = [i.item() for i in macro_averaged_accuracies_tensor]

    df = pd.DataFrame(macro_averaged_accuracies).transpose()

    # Create an ExcelWriter object
    with pd.ExcelWriter("macro_averaged_accuracy.xlsx") as writer:

        if task_type == "classification":
            df.index = ["Macro averaged accuracy (test set)"]
        
        df.columns = [f"Attention{i+1}" for i in range(number_of_attention_branches)]
        df.to_excel(writer)

        display(df)

    prediction_list = [i.to("cpu").detach().numpy().copy() for i in prediction_tensor_list]
    label_list = [i.to("cpu").detach().numpy().copy() for i in label_tensor_list]
    prediction_argmax_list = [i.to("cpu").detach().numpy().copy() for i in prediction_argmax_tensor_list]
    label_argmax_list = [i.to("cpu").detach().numpy().copy() for i in label_argmax_tensor_list]

    return prediction_list, label_list, prediction_argmax_list, label_argmax_list


def save_classification_report(
    prediction_argmax_list,
    label_argmax_list,
    task_type,
    number_of_attention_branches,
    class_name_list,
):

    if task_type == "classification":

        # Create an ExcelWriter object
        with pd.ExcelWriter("classification_report.xlsx") as writer:

            # Write each sheet to the Excel file
            for i in range(number_of_attention_branches):

                # Get classification report
                report = classification_report(
                    y_true=label_argmax_list[i], 
                    y_pred=prediction_argmax_list[i], 
                    target_names=class_name_list, 
                    output_dict=True,
                )
                
                report_df = pd.DataFrame(report).transpose()

                # Write the DataFrame to the Excel sheet
                report_df.to_excel(writer, sheet_name=f"Attention{i+1}")

                print("Attention", i+1)
                print(report_df)
                print(" ")


def save_prediction_tables(
    num_attentions: int,
    num_classes: int,
    post_analysis: Dict[str, Any],
    predictions: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Save prediction correctness and predicted class tables.
    Returns:
        pred_mask: Boolean array indicating correct predictions.
    """
    one_hot = np.eye(num_classes)

    def get_class_mask(class_idx: int) -> np.ndarray:
        target_one_hot = one_hot[class_idx]
        return np.all(post_analysis["dictionary"]["onehot_labels_array"] == target_one_hot, axis=1)

    def compute_prediction_mask(sample_mask: np.ndarray) -> np.ndarray:
        preds = predictions[sample_mask]
        lbls = labels[sample_mask]
        return preds == lbls
        

    with pd.ExcelWriter("prediction_true_false.xlsx") as writer_pred, \
            pd.ExcelWriter("prediction_class.xlsx") as writer_cls:
        
        pred_masks = []
        for class_idx, class_name in enumerate(post_analysis["class_name_list"]):
            print(f"Processing class '{class_name}'")
            mask = get_class_mask(class_idx)
            file_names = np.array(post_analysis["dictionary"]["file_names"])[mask].tolist()
            pred_mask = compute_prediction_mask(mask)        
            cols = [f"Attention{h+1}" for h in range(num_attentions)]

            # Correctness sheet
            df_correct = pd.DataFrame(pred_mask, columns=cols, index=file_names)
            df_correct.to_excel(writer_pred, sheet_name=class_name, index_label="File")

            # Predicted class sheet
            df_pred_cls = pd.DataFrame(predictions[mask], columns=cols, index=file_names)
            df_pred_cls.to_excel(writer_cls, sheet_name=class_name, index_label="File")

            pred_masks.append(pred_mask)

    return pred_masks



def sum_abs_diff_4d(arr, angle_feature_indices=None):
    """
    Given arr of shape (num_file, F, N, T), compute for each (file, feature, time)
    the sum of absolute differences across the N dimension.
    Values should be lower when a node’s feature values are similar to those of other nodes, and higher when they are different.

    For features listed in `angle_feature_indices`, values are treated as angles in radians,
    and wrap-around differences (e.g., 359° vs 1°) are handled appropriately.

    Parameters
    ----------
    arr : numpy.ndarray of shape (num_file, F, N, T)
        Input 4D array.

    angle_feature_indices : list or set of int
        Indices of features (along F dimension) to be treated as angles.

    Returns
    -------
    result : numpy.ndarray of shape (num_file, F, N, T)
        result[f, feat, i, t] = sum_{j=0..N-1} | arr[f, feat, i, t] - arr[f, feat, j, t] |.
    """
    num_file, F, N, T = arr.shape
    result = np.empty((num_file, F, N, T), dtype=np.float64)
    angle_feature_indices = set(angle_feature_indices or [])

    # Loop over file index, feature index, and time index
    for f_idx in range(num_file):
        for feat_idx in range(F):
            is_angle = feat_idx in angle_feature_indices
            for t_idx in range(T):
                # Extract 1D array along N dimension
                x = arr[f_idx, feat_idx, :, t_idx]

                # Compute pairwise difference matrix (N x N)
                diff = x[:, None] - x[None, :]

                if is_angle:
                    # Wrap differences into the range [-π, π]
                    diff = np.where(diff > np.pi, diff - 2 * np.pi, diff)
                    diff = np.where(diff < -np.pi, diff + 2 * np.pi, diff)

                # Sum of absolute differences for each element
                result[f_idx, feat_idx, :, t_idx] = np.sum(np.abs(diff), axis=1)

    return result


# ======================================================================
# Graph-based feature builders for model_type == "multi"
# ----------------------------------------------------------------------
# All builders return per-node features of shape (num_files, F_out, N, T)
# so they plug directly into CorrelationSaver / DistributionSaver configs.
# ======================================================================

def _normalize_edge_index_2d(edge_index_2d):
    """Return a (2, E) int array of 0-based node indices."""
    ei = np.asarray(edge_index_2d).astype(int)
    if ei.size > 0 and int(ei.min()) >= 1:
        ei = ei - 1
    return ei


def _build_incidence_lists(edge_index_2d, num_nodes):
    """node_id -> list of edge indices incident to that node."""
    _, E = edge_index_2d.shape
    incidence = [[] for _ in range(num_nodes)]
    for e in range(E):
        u = int(edge_index_2d[0, e])
        v = int(edge_index_2d[1, e])
        if 0 <= u < num_nodes:
            incidence[u].append(e)
        if 0 <= v < num_nodes and v != u:
            incidence[v].append(e)
    return incidence


def _find_distance_edge_index(edge_data_name_list):
    """Index of 'Distance between nodes' (or closest match) in edge_data_name_list, else None."""
    if not edge_data_name_list:
        return None
    for idx, name in enumerate(edge_data_name_list):
        if name == "Distance between nodes":
            return idx
    for idx, name in enumerate(edge_data_name_list):
        low = str(name).lower()
        if ("distance" in low) and ("node" in low):
            return idx
    return None


def _find_xy_indices(feature_name_list):
    """Return (x_idx, y_idx). Prefer corrected_x/corrected_y over x/y."""
    if "corrected_x" in feature_name_list and "corrected_y" in feature_name_list:
        return feature_name_list.index("corrected_x"), feature_name_list.index("corrected_y")
    if "x" in feature_name_list and "y" in feature_name_list:
        return feature_name_list.index("x"), feature_name_list.index("y")
    return None, None


def _find_head_direction_index(feature_name_list):
    for i, name in enumerate(feature_name_list):
        low = str(name).lower()
        if "head direction" in low:
            return i
    return None


def compute_incident_edge_features(edge_index, edge_data, num_nodes):
    """
    For each node i, aggregate edge_data values of the edges incident to i.
    Produces four statistics per original edge feature: sum, min, max, variance.

    Parameters
    ----------
    edge_index : np.ndarray of shape (num_files, 2, E)
    edge_data  : np.ndarray of shape (num_files, F_edge, E, T)
    num_nodes  : int

    Returns
    -------
    out   : np.ndarray of shape (num_files, F_edge * 4, N, T)
    order : [sum_{F}, min_{F}, max_{F}, var_{F}]
    """
    num_files, _, _ = edge_index.shape
    _, F_edge, _, T = edge_data.shape
    out = np.full((num_files, F_edge * 4, num_nodes, T), np.nan, dtype=np.float32)

    for f_idx in range(num_files):
        ei = _normalize_edge_index_2d(edge_index[f_idx])
        incidence = _build_incidence_lists(ei, num_nodes)

        for n in range(num_nodes):
            es = incidence[n]
            if not es:
                continue
            vals = edge_data[f_idx][:, es, :]  # (F_edge, k, T)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                all_nan = np.all(np.isnan(vals), axis=1)
                sum_ = np.where(all_nan, np.nan, np.nansum(vals, axis=1))
                min_ = np.nanmin(vals, axis=1)
                max_ = np.nanmax(vals, axis=1)
                var_ = np.nanvar(vals, axis=1)

            out[f_idx, 0 * F_edge : 1 * F_edge, n, :] = sum_
            out[f_idx, 1 * F_edge : 2 * F_edge, n, :] = min_
            out[f_idx, 2 * F_edge : 3 * F_edge, n, :] = max_
            out[f_idx, 3 * F_edge : 4 * F_edge, n, :] = var_

    return out


def incident_edge_feature_names(edge_data_name_list):
    sum_names = [f"{n} (incident sum)"      for n in edge_data_name_list]
    min_names = [f"{n} (incident min)"      for n in edge_data_name_list]
    max_names = [f"{n} (incident max)"      for n in edge_data_name_list]
    var_names = [f"{n} (incident variance)" for n in edge_data_name_list]
    return sum_names + min_names + max_names + var_names


def compute_mst_features(edge_index, edge_data, edge_data_name_list, num_nodes):
    """
    Per-node features extracted from the Minimum Spanning Tree (MST) built
    on the 'Distance between nodes' edge feature.

    Features (in order):
        0: MST degree
        1: MST is_leaf (1.0 if degree == 1, else 0.0)
        2: MST incident edge sum (node strength in MST)
        3: MST incident edge max (longest MST edge connected to the node)

    Returns
    -------
    out   : np.ndarray of shape (num_files, 4, N, T)
    names : list[str] of length 4
    """
    from scipy.sparse.csgraph import minimum_spanning_tree

    dist_idx = _find_distance_edge_index(edge_data_name_list)
    if dist_idx is None:
        raise ValueError(
            "Could not find 'Distance between nodes' in edge_data_name_list; "
            "MST features require a distance edge feature."
        )

    num_files, _, E = edge_index.shape
    _, _, _, T = edge_data.shape
    out = np.full((num_files, 4, num_nodes, T), np.nan, dtype=np.float32)

    for f_idx in range(num_files):
        ei = _normalize_edge_index_2d(edge_index[f_idx])
        u_arr = ei[0]
        v_arr = ei[1]

        for t in range(T):
            weights = edge_data[f_idx, dist_idx, :, t]
            if np.all(np.isnan(weights)):
                continue

            adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for e in range(E):
                w = weights[e]
                if np.isnan(w) or w <= 0:
                    continue
                u = int(u_arr[e])
                v = int(v_arr[e])
                aw = float(np.abs(w))
                # keep the smallest positive weight if the same pair appears twice
                if adj[u, v] == 0.0 or aw < adj[u, v]:
                    adj[u, v] = aw
                    adj[v, u] = aw

            if not np.any(adj > 0):
                continue

            mst = minimum_spanning_tree(adj).toarray()
            mst_sym = mst + mst.T

            incident_mask = mst_sym > 0
            degree = incident_mask.sum(axis=1).astype(np.float32)
            is_leaf = (degree == 1.0).astype(np.float32)
            incident_sum = mst_sym.sum(axis=1).astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                incident_max = np.where(
                    incident_mask.any(axis=1),
                    mst_sym.max(axis=1),
                    np.nan,
                ).astype(np.float32)

            out[f_idx, 0, :, t] = degree
            out[f_idx, 1, :, t] = is_leaf
            out[f_idx, 2, :, t] = incident_sum
            out[f_idx, 3, :, t] = incident_max

    names = [
        "MST degree",
        "MST is_leaf",
        "MST incident sum (distance)",
        "MST incident max (distance)",
    ]
    return out, names


def compute_geometric_features(features, feature_name_list, num_nodes):
    """
    Per-node geometric features derived from (x, y) coordinates.

    Features (in order):
        0: Distance to centroid
        1: Local density (Gaussian kernel, sigma = median pairwise distance per t)
        2: Head direction alignment (cos angle to circular-mean direction) -- only if head_direction is available

    Returns
    -------
    out   : np.ndarray of shape (num_files, n_out, N, T) or None if coordinates are unavailable
    names : list[str]
    """
    xi, yi = _find_xy_indices(feature_name_list)
    if xi is None or yi is None:
        return None, []

    hd_idx = _find_head_direction_index(feature_name_list)

    num_files = features.shape[0]
    _, _, _, T = features.shape
    n_out = 3 if hd_idx is not None else 2
    out = np.full((num_files, n_out, num_nodes, T), np.nan, dtype=np.float32)

    names = ["Distance to centroid", "Local density"]
    if hd_idx is not None:
        names.append("Head direction alignment")

    eye_mask = ~np.eye(num_nodes, dtype=bool)

    for f_idx in range(num_files):
        x = features[f_idx, xi, :, :].astype(np.float64)  # (N, T)
        y = features[f_idx, yi, :, :].astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cx = np.nanmean(x, axis=0)  # (T,)
            cy = np.nanmean(y, axis=0)

        dx = x - cx[None, :]
        dy = y - cy[None, :]
        out[f_idx, 0, :, :] = np.sqrt(dx * dx + dy * dy).astype(np.float32)

        for t in range(T):
            xt = x[:, t]
            yt = y[:, t]
            if np.all(np.isnan(xt)) or np.all(np.isnan(yt)):
                continue
            px = xt[:, None] - xt[None, :]
            py = yt[:, None] - yt[None, :]
            pd_mat = np.sqrt(px * px + py * py)
            off = pd_mat[eye_mask]
            off = off[~np.isnan(off) & (off > 0)]
            if off.size == 0:
                continue
            sigma = float(np.median(off))
            if sigma <= 0:
                continue
            kernel = np.exp(-(pd_mat * pd_mat) / (2.0 * sigma * sigma))
            np.fill_diagonal(kernel, 0.0)
            kernel = np.where(np.isnan(kernel), 0.0, kernel)
            out[f_idx, 1, :, t] = np.nansum(kernel, axis=1).astype(np.float32)

        if hd_idx is not None:
            hd = features[f_idx, hd_idx, :, :].astype(np.float64)  # radians
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_cos = np.nanmean(np.cos(hd), axis=0)
                mean_sin = np.nanmean(np.sin(hd), axis=0)
            mean_dir = np.arctan2(mean_sin, mean_cos)
            alignment = np.cos(hd - mean_dir[None, :])
            out[f_idx, 2, :, :] = alignment.astype(np.float32)

    return out, names


class _AttentionSaverBase:
    """Shared setup and class masking for the four saver classes.

    LOG_NAME is the name printed when a feature group is skipped.
    """

    LOG_NAME = "Saver"

    def __init__(
        self,
        post_analysis_dictionary: Dict[str, Any],
        attentions: np.ndarray,
        pred_masks: np.ndarray,
    ):
        # Store parameters as instance attributes
        self.post_dict = post_analysis_dictionary
        self.attentions = attentions
        self.pred_masks = pred_masks
        self.num_attentions = attentions.shape[1]
        self.num_nodes = attentions.shape[2]
        self.num_classes = post_analysis_dictionary["dictionary"]["onehot_labels_array"].shape[1]

        # Prepare one-hot identity matrix for class masking
        self.one_hot = np.eye(self.num_classes)

    def _get_class_mask(self, class_idx: int) -> np.ndarray:
        """
        Return a boolean mask for samples belonging to the given class index.
        """
        target_one_hot = self.one_hot[class_idx]
        return np.all(self.post_dict["dictionary"]["onehot_labels_array"] == target_one_hot, axis=1)


class _FeatureLoopSaverBase(_AttentionSaverBase):
    """Shared feature-set loop for CorrelationSaver and DistributionSaver.

    Both walk the same feature groups and differ only in where they write, so
    each subclass supplies its own naming through _writer_keys().
    """

    def __init__(
        self,
        model_type: str,
        post_analysis_dictionary: Dict[str, Any],
        attentions: np.ndarray,
        pred_masks: np.ndarray,
        angle_feature_indices,
    ):
        super().__init__(post_analysis_dictionary, attentions, pred_masks)
        self.model_type = model_type
        self.angle_feature_indices = angle_feature_indices

        # Heavy features (node_diff / edge / incident / MST / geometric) are cached
        # so _get_loop_configs is inexpensive when called once per class.
        self._cached_configs: List[Tuple[np.ndarray, List[str], str, str]] = None

    def _writer_keys(self, suffix: str) -> Tuple[str, str]:
        """Return (writer_key, highlighted_writer_key) for a feature group.

        Implemented by each subclass. suffix is one of "", "_node_diff",
        "_edge", "_edge_per_node", "_mst" or "_geometric".
        """
        raise NotImplementedError

    def _get_loop_configs(self) -> List[Tuple[np.ndarray, List[str], str, str]]:
        """
        Build configurations for each feature-set loop:
          each tuple is (features_array, feature_names, writer_key, highlighted_writer_key)
        """
        if self._cached_configs is not None:
            return self._cached_configs

        configs: List[Tuple[np.ndarray, List[str], str, str]] = []

        # Handcrafted features
        configs.append((
            self.post_dict["dictionary"]["features_for_post_analysis"],
            self.post_dict["feature_name_list"],
            *self._writer_keys(""),
        ))

        if self.model_type == "multi":
            # diff between nodes
            configs.append((
                sum_abs_diff_4d(self.post_dict["dictionary"]['features_for_post_analysis'], self.angle_feature_indices),
                self.post_dict["feature_name_list"],
                *self._writer_keys("_node_diff"),
            ))

            # average and variance over all edges (broadcast across nodes)
            ed_avg = np.mean(self.post_dict["dictionary"]["edge_data"], axis=2, keepdims=True)  # (num_file, F, 1, T)
            ed_var = np.var(self.post_dict["dictionary"]["edge_data"], axis=2, keepdims=True)  # (num_file, F, 1, T)
            ed_avg = np.repeat(ed_avg, repeats=self.num_nodes, axis=2)  # (num_file, F, N, T)
            ed_var = np.repeat(ed_var, repeats=self.num_nodes, axis=2)  # (num_file, F, N, T)
            if ed_avg is not None and ed_var is not None:
                stacked = np.concatenate([ed_avg, ed_var], axis=1)
                names_avg = [f"{n} (average)" for n in self.post_dict['edge_data_name_list']]
                names_var = [f"{n} (variance)" for n in self.post_dict['edge_data_name_list']]
                stacked_names = names_avg + names_var
                configs.append((
                    stacked,
                    stacked_names,
                    *self._writer_keys("_edge"),
                ))

            # NEW: incident edge aggregations per node (sum/min/max/variance)
            incident = compute_incident_edge_features(
                edge_index=self.post_dict["dictionary"]["edge_index"],
                edge_data=self.post_dict["dictionary"]["edge_data"],
                num_nodes=self.num_nodes,
            )
            configs.append((
                incident,
                incident_edge_feature_names(self.post_dict['edge_data_name_list']),
                *self._writer_keys("_edge_per_node"),
            ))

            # NEW: Minimum Spanning Tree features per node
            try:
                mst_feats, mst_names = compute_mst_features(
                    edge_index=self.post_dict["dictionary"]["edge_index"],
                    edge_data=self.post_dict["dictionary"]["edge_data"],
                    edge_data_name_list=self.post_dict["edge_data_name_list"],
                    num_nodes=self.num_nodes,
                )
                configs.append((
                    mst_feats,
                    mst_names,
                    *self._writer_keys("_mst"),
                ))
            except ValueError as exc:
                print(f"[{self.LOG_NAME}] Skipping MST features: {exc}")

            # NEW: geometric features from (x, y) coordinates
            geom_feats, geom_names = compute_geometric_features(
                features=self.post_dict["dictionary"]["features_for_post_analysis"],
                feature_name_list=self.post_dict["feature_name_list"],
                num_nodes=self.num_nodes,
            )
            if geom_feats is not None:
                configs.append((
                    geom_feats,
                    geom_names,
                    *self._writer_keys("_geometric"),
                ))
            else:
                print(f"[{self.LOG_NAME}] Skipping geometric features: (x, y) coordinates not found in feature_name_list.")

        self._cached_configs = configs
        return configs


class _ProximityGraphSaverBase(_AttentionSaverBase):
    """Shared distance-feature lookup for ClusterSaver and ConfigurationSaver."""

    def __init__(
        self,
        post_analysis_dictionary: Dict[str, Any],
        attentions: np.ndarray,
        pred_masks: np.ndarray,
        thresholds: List,
    ):
        super().__init__(post_analysis_dictionary, attentions, pred_masks)
        self.thresholds = thresholds
    def _get_distance_feature_index(self) -> int:
        """
        Find the feature index for "Distance between nodes" in edge_data_name_list.
        Uses a robust fallback (case-insensitive substring) if exact match not found.
        """
        names = self.post_dict.get("edge_data_name_list", [])
        if not names:
            raise ValueError("edge_data_name_list is missing in post_analysis_dictionary.")

        # Exact match first
        for idx, name in enumerate(names):
            if name == "Distance between nodes":
                return idx

        # Fallback: case-insensitive contains
        for idx, name in enumerate(names):
            low = str(name).lower()
            if ("distance" in low) and ("node" in low):
                return idx

        # If only one feature exists, assume it's the intended one
        if len(names) == 1:
            return 0

        raise ValueError(
            "Could not find 'Distance between nodes' in edge_data_name_list. "
            f"Available names: {names}"
        )


class CorrelationSaver(_FeatureLoopSaverBase):
    """
    Class for saving feature correlation tables into Excel files.
    """

    LOG_NAME = "CorrelationSaver"

    def _writer_keys(self, suffix: str) -> Tuple[str, str]:
        return f"correlation{suffix}", f"correlation{suffix}_in_highlighted_segments"

    def compute_correlations(
        self,
        att_class: np.ndarray,
        features: np.ndarray,
        pred_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute correlations between attention weights and features.
        Returns two arrays of shape (num_files, num_attentions, num_features):
          - overall correlations
          - highlighted-region correlations
        """

        num_files, num_attentions, num_nodes, _ = att_class.shape
        num_features = features.shape[1]
        corr = np.full((num_files, num_attentions, num_features), np.nan)
        corr_hl = np.full((num_files, num_attentions, num_features), np.nan)

        for i in range(num_files):

            # Identify valid timestamps (not all-NaN)
            valid_attentions = np.where(pred_mask[i])[0]
            F = features[i]  # (num_features, num_nodes, num_timestamps)
            valid_ts = ~np.all(np.isnan(F), axis=(0,1))
            if not valid_ts.any():
                continue

            # Flatten features with two-step slicing to avoid incorrect indexing
            Y_masked = np.compress(valid_ts, F, axis=2)       # (num_features, num_nodes, valid_count)
            Y_flat   = Y_masked.reshape(num_features, -1)
            Y_mean   = Y_flat.mean(axis=1)
            Y_ctr    = Y_flat - Y_mean[:, None]
            Y_std    = np.sqrt((Y_ctr**2).sum(axis=1))

            threshold = 1.0 / (num_nodes * valid_ts.sum())

            for h in valid_attentions:
                x2d    = att_class[i, h]                       # (num_nodes, num_timestamps)
                x_mask = x2d[:, valid_ts]                       # (num_nodes, valid_count)
                x_flat = x_mask.ravel()
                x_mean = x_flat.mean()
                x_ctr  = x_flat - x_mean
                x_std  = np.sqrt((x_ctr**2).sum())

                # overall correlation
                denom = x_std * Y_std
                valid = denom != 0
                vals = np.full(num_features, np.nan)
                if valid.any():
                    nums = (Y_ctr * x_ctr).sum(axis=1)
                    vals[valid] = nums[valid] / denom[valid]
                corr[i, h, :] = vals

                # highlighted-region correlation
                hl_mask = x_flat >= threshold
                if hl_mask.any():
                    x_hl     = x_flat[hl_mask]
                    Y_hl     = Y_flat[:, hl_mask]
                    x_hl_ctr = x_hl - x_hl.mean()
                    Y_hl_ctr = Y_hl - Y_hl.mean(axis=1)[:, None]
                    x_hl_std = np.sqrt((x_hl_ctr**2).sum())
                    Y_hl_std = np.sqrt((Y_hl_ctr**2).sum(axis=1))
                    denom_hl = x_hl_std * Y_hl_std

                    vals_hl = np.full(num_features, np.nan)
                    valid_hl = denom_hl != 0
                    if valid_hl.any():
                        nums_hl = (Y_hl_ctr * x_hl_ctr).sum(axis=1)
                        vals_hl[valid_hl] = nums_hl[valid_hl] / denom_hl[valid_hl]
                    corr_hl[i, h, :] = vals_hl

        return corr, corr_hl

    def write_correlation_sheets(
        self,
        class_name: str,
        corr: np.ndarray,
        corr_hl: np.ndarray,
        feature_names: List[str],
        writer_key: str,
        hl_writer_key: str
    ):
        """
        Average correlations over samples and write to the specified sheets.
        """

        # Average over the sample axis and transpose for DataFrame
        avg = np.nanmean(corr, axis=0).T
        avg_hl = np.nanmean(corr_hl, axis=0).T
        _, num_attentions = avg.shape

        cols = [f"Attention{h+1}" for h in range(num_attentions)]
        df = pd.DataFrame(avg, index=feature_names, columns=cols)
        df_hl = pd.DataFrame(avg_hl, index=feature_names, columns=cols)

        path = f"{writer_key}.xlsx"
        mode = "a" if os.path.exists(path) else "w"
        with pd.ExcelWriter(f"{writer_key}.xlsx", mode=mode) as writer, \
             pd.ExcelWriter(f"{hl_writer_key}.xlsx", mode=mode) as hl_writer:

            df.to_excel(writer, sheet_name=class_name, index_label="Feature")
            df_hl.to_excel(hl_writer, sheet_name=class_name, index_label="Feature")

    def save_all(self):
        """
        Main entry point: loop over each class, write prediction and correlation tables,
        then close all writers.
        """
        for class_idx, class_name in enumerate(self.post_dict["class_name_list"]):
            print(f"Start processing class '{class_name}'")
            mask = self._get_class_mask(class_idx)

            # Slice attention and masks for this class
            att_class = self.attentions[mask]
            pred_mask_class = self.pred_masks[class_idx]

            for features, feat_names, w_key, hl_key in self._get_loop_configs():
                feat_class = features[mask]
                corr, corr_hl = self.compute_correlations(att_class, feat_class, pred_mask_class)
                self.write_correlation_sheets(class_name, corr, corr_hl, feat_names, w_key, hl_key)


class DistributionSaver(_FeatureLoopSaverBase):
    """
    Build and save histogram dictionaries and hellinger_distance.xlsx of highlighted feature segments.
    """

    LOG_NAME = "DistributionSaver"

    def _writer_keys(self, suffix: str) -> Tuple[str, str]:
        return f"histogram{suffix}.pkl", f"hellinger_distance{suffix}"

    def aggregate_highlighted_segments(self, feats_cls, atts_cls, pred_cls):
        """
        For each attention and feature, collect all feature‐values
        at positions where the attention weight exceeds the uniform threshold,
        but only in correctly predicted samples.

        Returns:
            aggregated[attention][feat] = list of values
        """

        num_files, num_feats, num_nodes, _ = feats_cls.shape  # shape (n_f, n_feat, N, T)
        num_attentions = atts_cls.shape[1]  # shape (n_f, n_attentions, N, T)

        # Initialize aggregation: aggregated[attention_id][feature_id] = list of values
        aggregated = [[[] for _ in range(num_feats)] for _ in range(num_attentions)]

        for i in range(num_files):
            # features_class[i]: shape = (num_features, N, T)
            feat_i = feats_cls[i]     # (num_feats, N, T)
            # Get and flatten the attention map for the current file
            att_i  = atts_cls[i]      # (num_attentions, N, T)

            # Extract valid timestamps
            valid_t = ~np.all(np.isnan(feat_i), axis=(0,1))  # shape=(T,)
            n_valid = valid_t.sum()
            if n_valid == 0:
                continue

            # Calculate the attention threshold
            threshold = 1.0 / (num_nodes * n_valid)

            for j in range(num_attentions):
                # Process only if the prediction is correct for this attention
                if not pred_cls[i, j]:
                    continue
                att_flat  = att_i[j, :, valid_t].ravel()  # length = N * num_valid_ts
                for k in range(num_feats):
                    # Get and flatten the feature map for the current feature
                    feat_flat = feat_i[k, :, valid_t].ravel()
                    # Select only the highlighted segments based on the threshold
                    mask_hl   = att_flat >= threshold
                    aggregated[j][k].extend(feat_flat[mask_hl])

        return aggregated

    def build_dicts(self, per_class_aggs, names: Sequence[str]) -> Dict:
        """
        Build both dictionaries for all class‐pairs, storing raw arrays
        (arr_i, arr_j) under keys like:
            "attention{h+1} condition {i+1} - {j+1}" → { feature_name: {i: arr_i, j: arr_j} }
        """

        num_classes = len(per_class_aggs)
        num_attentions = len(per_class_aggs[0])
        distribution_dict = {}
        hellinger_distance_dict = {}

        for h in range(num_attentions):
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    key = f"Attention{h+1} Condition {i+1} - {j+1}"
                    bucket_dist = distribution_dict.setdefault(key, {})
                    bucket_hellinger = hellinger_distance_dict.setdefault(key, {})
                    for idx, feat_name in enumerate(names):
                        arr_i = np.array(per_class_aggs[i][h][idx])
                        arr_j = np.array(per_class_aggs[j][h][idx])

                        # Keep distribution
                        bucket_dist.setdefault(feat_name, {})[i] = arr_i
                        bucket_dist.setdefault(feat_name, {})[j] = arr_j

                        # Keep Hellinger distance
                        if arr_i.size == 0 or arr_j.size == 0:
                            print("Warning: the input array is empty! Attention", h, "class", i, "- class", j)
                            distance = np.nan
                        elif np.isnan(arr_i).any() or np.isnan(arr_j).any():
                            print("Warning: NaN values detected in input arrays! Attention", h, "class", i, "- class", j)
                            distance = np.nan
                        elif np.isinf(arr_i).any() or np.isinf(arr_j).any():
                            print("Warning: Infinite values detected in input arrays! Attention", h, "class", i, "- class", j)
                            distance = np.nan
                        else:
                            # Compute histograms for both arrays using common bin edges
                            combined = np.concatenate([arr_i, arr_j])
                            bin_edges = np.histogram_bin_edges(combined, bins=50)
                            hist_i_counts, _ = np.histogram(arr_i, bins=bin_edges)
                            hist_j_counts, _ = np.histogram(arr_j, bins=bin_edges)
                            # Normalize counts to get probability distributions
                            p = hist_i_counts / np.sum(hist_i_counts) if np.sum(hist_i_counts) > 0 else np.zeros_like(hist_i_counts)
                            q = hist_j_counts / np.sum(hist_j_counts) if np.sum(hist_j_counts) > 0 else np.zeros_like(hist_j_counts)
                            # Compute Hellinger distance:
                            # H(P, Q) = (1/sqrt(2)) * sqrt(sum((sqrt(p) - sqrt(q))^2))
                            distance = np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2)) / np.sqrt(2)
                        bucket_hellinger[feat_name] = distance
        
        return distribution_dict, hellinger_distance_dict

    def save_all(self) -> None:
        configs = self._get_loop_configs()
        for features, names, dict_save_name, xlsx_save_name in configs:

            per_class_aggs = []
            for cls in range(self.num_classes):
                # Select files corresponding to the given class
                mask = self._get_class_mask(cls)
                feats_cls = features[mask]        # shape (n_f, n_feat, N, T)
                atts_cls = self.attentions[mask]  # shape (n_f, n_attentions, N, T)
                pred_cls = self.pred_masks[cls]
                agg = self.aggregate_highlighted_segments(feats_cls, atts_cls, pred_cls)
                per_class_aggs.append(agg)

            distribution_dict, hellinger_distance_dict = self.build_dicts(per_class_aggs, names)

            # Save distribution
            with gzip.open(dict_save_name, "wb") as f:
                pickle.dump(distribution_dict, f)

            # Save Hellinger distance
            path = f"{xlsx_save_name}.xlsx"
            mode = "a" if os.path.exists(path) else "w"
            with pd.ExcelWriter(f"{xlsx_save_name}.xlsx", mode=mode) as writer:
                df = pd.DataFrame(hellinger_distance_dict)
                df.to_excel(writer, index_label="Feature")


class ClusterSaver(_ProximityGraphSaverBase):
    """
    Class for saving correlation tables (attention vs. cluster-count indicators) into Excel files.
    - Uses ONLY "Distance between nodes" edge feature to compute cluster counts.
    - Builds binary features for cluster count x (x=1..N) and computes correlations.
    - Saves per-class sheets into:
        correlation_cluster.xlsx
        correlation_cluster_in_highlighted_segments.xlsx
    """

    def _count_clusters(self, edge_index, edge_weight, number_of_nodes, thresh):
        """
        Count number of connected components (clusters) per (file, feature, time),
        using threshold filtering on edge weights.
        Returns shape (num_file, F, N, T) after repeating along node axis.
        """
        num_file, _, E = edge_index.shape
        num_file2, F, E2, T = edge_weight.shape
        assert num_file == num_file2, "Number of files in edge_index and edge_weight do not match"
        assert E == E2, "Number of edges in edge_index and edge_weight do not match"

        cluster_counts = np.full((num_file, F, 1, T), np.nan, dtype=float)

        for f_idx in range(num_file):
            for feat_idx in range(F):
                for t_idx in range(T):
                    edges = edge_index[f_idx]  # (2, E)
                    weights = edge_weight[f_idx, feat_idx, :, t_idx]  # (E,)
                    if np.all(np.isnan(weights)):
                        continue

                    adj = defaultdict(list)
                    for i in range(edges.shape[1]):
                        u, v = edges[0, i], edges[1, i]
                        w = weights[i]
                        if w <= thresh[feat_idx]:
                            adj[u].append(v)
                            adj[v].append(u)

                    visited = set()
                    clusters = []
                    for node in range(number_of_nodes):
                        if node in visited:
                            continue
                        queue = deque([node])
                        visited.add(node)
                        component = []
                        while queue:
                            u = queue.popleft()
                            component.append(u)
                            for neighbor in adj[u]:
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    queue.append(neighbor)
                        clusters.append(component)

                    cluster_counts[f_idx, feat_idx, 0, t_idx] = len(clusters)

        # Repeat to shape (Files, F, N, T) to match downstream expectations
        cluster_counts = np.repeat(cluster_counts, repeats=number_of_nodes, axis=2)
        return cluster_counts

    def _build_cluster_count_indicators(self, cluster_counts_1feat: np.ndarray) -> np.ndarray:
        """
        cluster_counts_1feat: (num_files, 1, N_nodes, T)
        Build indicator features for x=1..N:
          features_bin: (num_files, N, N_nodes, T)
        where:
          - 1 if cluster_count == x
          - 0 otherwise
          - NaN where cluster_count is NaN ("cluster not exists")
        """
        # base: (num_files, N_nodes, T)  (counts are repeated across node axis anyway)
        base = cluster_counts_1feat[:, 0, :, :]
        num_files, n_nodes, T = base.shape
        N = self.num_nodes  # maximum possible clusters

        feats = np.full((num_files, N, n_nodes, T), np.nan, dtype=float)

        # Keep NaN where base is NaN, else set 0/1
        base_is_nan = np.isnan(base)
        for x in range(1, N + 1):
            bin_mat = np.where(base_is_nan, np.nan, (base == float(x)).astype(float))
            feats[:, x - 1, :, :] = bin_mat

        return feats

    def _write_cluster_correlation_sheets(
        self,
        class_name: str,
        corr: np.ndarray,
        corr_hl: np.ndarray,
        N: int,
        writer_key: str = "correlation_cluster",
        hl_writer_key: str = "correlation_cluster_in_highlighted_segments",
    ):
        """
        Save per-class sheet with:
          - No "Feature" index/column
          - First column: "Feature" (1..N)
          - Then columns: Attention1..AttentionH
        Values are averaged over samples (files).
        """
        # Average over samples/files: (num_attentions, N) then transpose -> (N, num_attentions)
        avg = np.nanmean(corr, axis=0).T
        avg_hl = np.nanmean(corr_hl, axis=0).T

        # Columns: Attention1..AttentionH
        _, num_attentions = avg.shape
        cols = [f"Attention{h+1}" for h in range(num_attentions)]

        df = pd.DataFrame(avg, columns=cols)
        df.insert(0, "Feature", np.arange(1, N + 1, dtype=int))

        df_hl = pd.DataFrame(avg_hl, columns=cols)
        df_hl.insert(0, "Feature", np.arange(1, N + 1, dtype=int))

        # Write overall
        path = f"{writer_key}.xlsx"
        mode = "a" if os.path.exists(path) else "w"
        writer_kwargs = {"engine": "openpyxl", "mode": mode}
        if mode == "a":
            writer_kwargs["if_sheet_exists"] = "replace"  # rerun-safe

        with pd.ExcelWriter(path, **writer_kwargs) as writer:
            df.to_excel(writer, sheet_name=class_name, index=False)

        # Write highlighted
        path_hl = f"{hl_writer_key}.xlsx"
        mode_hl = "a" if os.path.exists(path_hl) else "w"
        writer_kwargs_hl = {"engine": "openpyxl", "mode": mode_hl}
        if mode_hl == "a":
            writer_kwargs_hl["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(path_hl, **writer_kwargs_hl) as writer:
            df_hl.to_excel(writer, sheet_name=class_name, index=False)

    def save_all(self):
        """
        Main entry point:
          - For each class:
            - compute cluster counts using ONLY "Distance between nodes"
            - build indicator features for #clusters x=1..N
            - compute correlations via CorrelationSaver.compute_correlations
            - save into correlation_cluster.xlsx and correlation_cluster_in_highlighted_segments.xlsx
        """
        # Find "Distance between nodes" feature index once
        dist_idx = self._get_distance_feature_index()

        thresholds_arr = np.asarray(self.thresholds)
        if thresholds_arr.ndim != 1:
            thresholds_arr = thresholds_arr.reshape(-1)
        if dist_idx >= len(thresholds_arr):
            raise ValueError(
                f"thresholds length ({len(thresholds_arr)}) is smaller than dist_idx ({dist_idx})."
            )
        dist_thresh = thresholds_arr[dist_idx: dist_idx + 1]  # shape (1,)

        for class_idx, class_name in enumerate(self.post_dict["class_name_list"]):
            print(f"Start processing class '{class_name}'")

            mask = self._get_class_mask(class_idx)
            pred_mask = self.pred_masks[class_idx]  # expected to align with att_class samples
            att_class = self.attentions[mask]

            # Slice edge data to ONLY distance feature
            edge_index_cls = self.post_dict["dictionary"]["edge_index"][mask]
            edge_data_cls = self.post_dict["dictionary"]["edge_data"][mask]  # (files, F, E, T)
            edge_data_dist = np.abs(edge_data_cls[:, dist_idx:dist_idx + 1, :, :])  # (files, 1, E, T)

            # Count clusters (Distance between nodes only)
            cluster_counts = self._count_clusters(
                edge_index=edge_index_cls,
                edge_weight=edge_data_dist,
                number_of_nodes=self.num_nodes,
                thresh=dist_thresh,
            )  # (files, 1, N_nodes, T)

            # Build binary indicator features for x=1..N
            features_bin = self._build_cluster_count_indicators(cluster_counts)  # (files, N, N_nodes, T)

            # Correlations (reuse CorrelationSaver)
            corr, corr_hl = CorrelationSaver.compute_correlations(
                None,
                att_class=att_class,
                features=features_bin,
                pred_mask=pred_mask,
            )

            # Save (per class)
            self._write_cluster_correlation_sheets(
                class_name=class_name,
                corr=corr,
                corr_hl=corr_hl,
                N=self.num_nodes,
                writer_key="correlation_cluster",
                hl_writer_key="correlation_cluster_in_highlighted_segments",
            )


class ConfigurationSaver(_ProximityGraphSaverBase):
    """
    Class for saving correlation tables (attention vs. spatial-configuration indicators) into Excel files.
    - Uses ONLY the "Distance between nodes" edge feature. At each timepoint it builds the proximity
      graph in which two nodes are connected when their distance is within the threshold.
    - That proximity graph is classified into an unlabeled-graph isomorphism class (a "configuration"),
      i.e. the connection pattern considered up to node relabeling. Binary indicator features are built
      for each configuration id and correlated with attention.
    - This generalizes the 4-node analysis (11 configurations) to any node count, but is only enabled
      for models with MAX_NODES (=5) or fewer nodes; beyond that the number of possible configurations
      explodes (2 nodes -> 2, 3 -> 4, 4 -> 11, 5 -> 34, 6 -> 156, ...), which is impractical to interpret.
    - Saves per-class sheets into:
        correlation_config.xlsx
        correlation_config_in_highlighted_segments.xlsx
    """

    MAX_NODES = 5

    def __init__(
        self,
        post_analysis_dictionary: Dict[str, Any],
        attentions: np.ndarray,
        pred_masks: np.ndarray,
        thresholds: List,
    ):
        super().__init__(post_analysis_dictionary, attentions, pred_masks, thresholds)

        # Ordered list of node pairs (i < j) used to pack/unpack the proximity-graph bit pattern.
        self.pairs = [(i, j) for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes)]

        # Map every labeled proximity-graph bit pattern to a configuration id (only feasible for small N).
        if self.num_nodes <= self.MAX_NODES:
            self.bitpattern_to_configid, self.num_configs = self._build_configuration_lookup(
                self.num_nodes, self.pairs
            )
        else:
            self.bitpattern_to_configid, self.num_configs = None, 0
    @staticmethod
    def _build_configuration_lookup(number_of_nodes, pairs):
        """
        Build a lookup mapping every labeled proximity-graph bit pattern (0 .. 2**len(pairs)-1) to a
        configuration id (0-based). Two patterns share an id iff they are isomorphic, i.e. equal up to
        node relabeling.

        Bits are packed left-to-right in `pairs` order: pairs[0] is the most significant bit.
        Configuration ids are assigned in ascending order of each class's canonical (minimum) pattern,
        so the empty graph is always id 0 and the assignment is deterministic.

        Returns:
            lookup       : np.ndarray of shape (2**len(pairs),), lookup[b] = configuration id of pattern b
            num_configs  : number of distinct configurations (= number of unlabeled graphs on N nodes)
        """
        import itertools

        num_pairs = len(pairs)
        pair_index = {pair: k for k, pair in enumerate(pairs)}

        def canonical(b):
            best = None
            for perm in itertools.permutations(range(number_of_nodes)):
                val = 0
                for (i, j) in pairs:
                    u, v = perm[i], perm[j]
                    if u > v:
                        u, v = v, u
                    k = pair_index[(u, v)]
                    bit = (b >> (num_pairs - 1 - k)) & 1
                    val = (val << 1) | bit
                if best is None or val < best:
                    best = val
            return best

        rep_to_cid: Dict[int, int] = {}
        lookup = np.empty(1 << num_pairs, dtype=int)
        for b in range(1 << num_pairs):
            rep = canonical(b)
            if rep not in rep_to_cid:
                rep_to_cid[rep] = len(rep_to_cid)
            lookup[b] = rep_to_cid[rep]

        return lookup, len(rep_to_cid)

    def _classify_configurations(self, edge_index, edge_weight, number_of_nodes, thresh):
        """
        For each (file, feature, time) build the proximity graph from the distance feature and return its
        configuration id. A node pair is connected when it is present in edge_index and its distance is
        <= thresh. Pairs not present (or above the threshold) are treated as disconnected, mirroring
        ClusterSaver._count_clusters.

        Returns shape (num_file, F, N, T) after repeating along the node axis; NaN where the distance
        feature is entirely missing at that timepoint.
        """
        num_file, _, E = edge_index.shape
        num_file2, F, E2, T = edge_weight.shape
        assert num_file == num_file2, "Number of files in edge_index and edge_weight do not match"
        assert E == E2, "Number of edges in edge_index and edge_weight do not match"

        num_pairs = len(self.pairs)
        config_ids = np.full((num_file, F, 1, T), np.nan, dtype=float)

        for f_idx in range(num_file):
            edges = edge_index[f_idx]  # (2, E)
            for feat_idx in range(F):
                for t_idx in range(T):
                    weights = edge_weight[f_idx, feat_idx, :, t_idx]  # (E,)
                    if np.all(np.isnan(weights)):
                        continue

                    # Smallest distance per undirected pair (a pair may appear as both (u,v) and (v,u)).
                    present: Dict[Tuple[int, int], float] = {}
                    for e in range(E):
                        u, v = int(edges[0, e]), int(edges[1, e])
                        if u == v:
                            continue
                        if u > v:
                            u, v = v, u
                        w = weights[e]
                        if np.isnan(w):
                            continue
                        if (u, v) not in present or w < present[(u, v)]:
                            present[(u, v)] = w

                    # Pack one bit per node pair (1 if connected within the threshold, else 0).
                    b = 0
                    for (i, j) in self.pairs:
                        w = present.get((i, j), None)
                        bit = 1 if (w is not None and w <= thresh[feat_idx]) else 0
                        b = (b << 1) | bit

                    config_ids[f_idx, feat_idx, 0, t_idx] = self.bitpattern_to_configid[b]

        # Repeat to shape (Files, F, N, T) to match downstream expectations
        config_ids = np.repeat(config_ids, repeats=number_of_nodes, axis=2)
        return config_ids

    def _build_config_indicators(self, config_ids_1feat: np.ndarray) -> np.ndarray:
        """
        config_ids_1feat: (num_files, 1, N_nodes, T)
        Build indicator features for each configuration id k (0..num_configs-1):
          features_bin: (num_files, num_configs, N_nodes, T)
        where:
          - 1 if config_id == k
          - 0 otherwise
          - NaN where config_id is NaN ("configuration not available")
        """
        base = config_ids_1feat[:, 0, :, :]  # (num_files, N_nodes, T)
        num_files, n_nodes, T = base.shape
        K = self.num_configs

        feats = np.full((num_files, K, n_nodes, T), np.nan, dtype=float)

        base_is_nan = np.isnan(base)
        for k in range(K):
            bin_mat = np.where(base_is_nan, np.nan, (base == float(k)).astype(float))
            feats[:, k, :, :] = bin_mat

        return feats

    def _write_config_correlation_sheets(
        self,
        class_name: str,
        corr: np.ndarray,
        corr_hl: np.ndarray,
        K: int,
        writer_key: str = "correlation_config",
        hl_writer_key: str = "correlation_config_in_highlighted_segments",
    ):
        """
        Save per-class sheet with:
          - First column: "Feature" (configuration id, 1..K)
          - Then columns: Attention1..AttentionH
        Values are averaged over samples (files).
        """
        avg = np.nanmean(corr, axis=0).T
        avg_hl = np.nanmean(corr_hl, axis=0).T

        _, num_attentions = avg.shape
        cols = [f"Attention{h+1}" for h in range(num_attentions)]

        df = pd.DataFrame(avg, columns=cols)
        df.insert(0, "Feature", np.arange(1, K + 1, dtype=int))

        df_hl = pd.DataFrame(avg_hl, columns=cols)
        df_hl.insert(0, "Feature", np.arange(1, K + 1, dtype=int))

        # Write overall
        path = f"{writer_key}.xlsx"
        mode = "a" if os.path.exists(path) else "w"
        writer_kwargs = {"engine": "openpyxl", "mode": mode}
        if mode == "a":
            writer_kwargs["if_sheet_exists"] = "replace"  # rerun-safe

        with pd.ExcelWriter(path, **writer_kwargs) as writer:
            df.to_excel(writer, sheet_name=class_name, index=False)

        # Write highlighted
        path_hl = f"{hl_writer_key}.xlsx"
        mode_hl = "a" if os.path.exists(path_hl) else "w"
        writer_kwargs_hl = {"engine": "openpyxl", "mode": mode_hl}
        if mode_hl == "a":
            writer_kwargs_hl["if_sheet_exists"] = "replace"

        with pd.ExcelWriter(path_hl, **writer_kwargs_hl) as writer:
            df_hl.to_excel(writer, sheet_name=class_name, index=False)

    def plot_configuration_legend(self, save_path: str = None):
        """
        Draw the representative proximity graph for each configuration id (labeled 1..num_configs) so the
        correlation tables can be interpreted. Nodes are placed on a circle; an edge means the two nodes
        are within the distance threshold. Returns silently for unsupported node counts.
        """
        if self.num_nodes > self.MAX_NODES or self.num_configs == 0:
            print(
                f"[ConfigurationSaver] Configuration legend is available only for models with "
                f"{self.MAX_NODES} or fewer nodes (this model has {self.num_nodes} nodes)."
            )
            return

        # Representative (smallest) bit pattern for each configuration id.
        rep_pattern: Dict[int, int] = {}
        for b in range(len(self.bitpattern_to_configid)):
            cid = int(self.bitpattern_to_configid[b])
            if cid not in rep_pattern:
                rep_pattern[cid] = b

        K = self.num_configs
        ncols = min(6, K)
        nrows = int(np.ceil(K / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.6 * nrows))
        axes = np.atleast_1d(axes).ravel()

        # Node positions on a circle (node 1 at the top, going clockwise).
        angles = np.pi / 2 - np.linspace(0, 2 * np.pi, self.num_nodes, endpoint=False)
        pos = np.column_stack([np.cos(angles), np.sin(angles)])
        num_pairs = len(self.pairs)

        for cid in range(K):
            ax = axes[cid]
            b = rep_pattern[cid]
            for k, (i, j) in enumerate(self.pairs):
                bit = (b >> (num_pairs - 1 - k)) & 1
                if bit:
                    ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                            "-", color="gray", linewidth=1.5, zorder=1)
            ax.scatter(pos[:, 0], pos[:, 1], s=350, c="lightblue", edgecolors="k", zorder=2)
            for n in range(self.num_nodes):
                ax.text(pos[n, 0], pos[n, 1], str(n + 1), ha="center", va="center", zorder=3)
            ax.set_title(f"Config. {cid + 1}")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)

        for idx in range(K, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_all(self):
        """
        Main entry point:
          - Skip entirely for models with more than MAX_NODES nodes.
          - For each class:
            - classify the proximity graph at every timepoint into a configuration id
              (using ONLY "Distance between nodes")
            - build indicator features for each configuration id
            - compute correlations via CorrelationSaver.compute_correlations
            - save into correlation_config.xlsx and correlation_config_in_highlighted_segments.xlsx
        """
        if self.num_nodes > self.MAX_NODES:
            print(
                f"[ConfigurationSaver] Skipped: configurations are computed only for models with "
                f"{self.MAX_NODES} or fewer nodes (this model has {self.num_nodes} nodes)."
            )
            return

        # Find "Distance between nodes" feature index once
        dist_idx = self._get_distance_feature_index()

        thresholds_arr = np.asarray(self.thresholds)
        if thresholds_arr.ndim != 1:
            thresholds_arr = thresholds_arr.reshape(-1)
        if dist_idx >= len(thresholds_arr):
            raise ValueError(
                f"thresholds length ({len(thresholds_arr)}) is smaller than dist_idx ({dist_idx})."
            )
        dist_thresh = thresholds_arr[dist_idx: dist_idx + 1]  # shape (1,)

        for class_idx, class_name in enumerate(self.post_dict["class_name_list"]):
            print(f"Start processing class '{class_name}'")

            mask = self._get_class_mask(class_idx)
            pred_mask = self.pred_masks[class_idx]  # expected to align with att_class samples
            att_class = self.attentions[mask]

            # Slice edge data to ONLY distance feature
            edge_index_cls = self.post_dict["dictionary"]["edge_index"][mask]
            edge_data_cls = self.post_dict["dictionary"]["edge_data"][mask]  # (files, F, E, T)
            edge_data_dist = np.abs(edge_data_cls[:, dist_idx:dist_idx + 1, :, :])  # (files, 1, E, T)

            # Classify configurations (Distance between nodes only)
            config_ids = self._classify_configurations(
                edge_index=edge_index_cls,
                edge_weight=edge_data_dist,
                number_of_nodes=self.num_nodes,
                thresh=dist_thresh,
            )  # (files, 1, N_nodes, T)

            # Build binary indicator features for each configuration id
            features_bin = self._build_config_indicators(config_ids)  # (files, num_configs, N_nodes, T)

            # Correlations (reuse CorrelationSaver)
            corr, corr_hl = CorrelationSaver.compute_correlations(
                None,
                att_class=att_class,
                features=features_bin,
                pred_mask=pred_mask,
            )

            # Save (per class)
            self._write_config_correlation_sheets(
                class_name=class_name,
                corr=corr,
                corr_hl=corr_hl,
                K=self.num_configs,
                writer_key="correlation_config",
                hl_writer_key="correlation_config_in_highlighted_segments",
            )


def mean_node_attention(
    post_analysis_dictionary,
    attentions,
    pred_masks,
):

    num_attentions = attentions.shape[1]
    num_classes = post_analysis_dictionary["dictionary"]["onehot_labels_array"].shape[1]

    def _get_class_mask(class_idx: int) -> np.ndarray:
        """
        Return a boolean mask for samples belonging to the given class index.
        """
        target_one_hot = np.eye(num_classes)[class_idx]
        return np.all(post_analysis_dictionary["dictionary"]["onehot_labels_array"] == target_one_hot, axis=1)

    with pd.ExcelWriter("node_attention.xlsx") as writer:
        for class_idx in range(num_classes):
            mask = _get_class_mask(class_idx)
            att_class = attentions[mask]
            att_class = np.mean(att_class, axis=3)
            pred_class = pred_masks[class_idx]
            pred_class = np.expand_dims(pred_class, axis=2)

            result = (att_class * pred_class).sum(axis=0)

            total = result.sum()
            if total != 0:
                col_sums = result.sum(axis=1, keepdims=True)
                normalized = result / col_sums    
            else:
                normalized = np.zeros_like(result)

            df = pd.DataFrame(normalized).transpose()
            _, num_nodes = result.shape
            df.index = [f"Node{i+1}" for i in range(num_nodes)]
            df.columns = [f"Attention{i+1}" for i in range(num_attentions)]
            sheet_name = post_analysis_dictionary["class_name_list"][class_idx]
            df.to_excel(writer, sheet_name=sheet_name)


def save_data_for_interactive_attention_plots(
    model_type,
    number_of_classes,
    has_pose,
    post_analysis_dictionary,
    attentions,
    predictions_argmax,
    labels_argmax,
):
    
    one_hot = np.eye(number_of_classes)  # One-hot encoding for each class
    feature_name_list = post_analysis_dictionary["feature_name_list"]
    class_name_list = post_analysis_dictionary["class_name_list"]

    new_dictionary = {}
    for i in range(number_of_classes):

        mask_class = np.all((post_analysis_dictionary["dictionary"]["onehot_labels_array"] == one_hot[i]), axis=1)  # (number_of_files_of_all_classes)
        handcrafted_features = post_analysis_dictionary["dictionary"]["features_for_post_analysis"][mask_class]

        time_index = [feature_name_list.index("time")]
        time_to_draw_map = handcrafted_features[:, time_index, :, :]  # (number_of_files, 1, N, T)

        if "corrected_x" in feature_name_list:
            xy_indices = [feature_name_list.index("corrected_x"), feature_name_list.index("corrected_y")]
        else:
            xy_indices = [feature_name_list.index("x"), feature_name_list.index("y")]
        xy_to_draw_map = handcrafted_features[:, xy_indices, :, :]  # (number_of_files, 2, N, T)
        draw_map = "xy"

        if (model_type == "single") and ("latitude" in feature_name_list) and ("longitude" in feature_name_list):
            latlon_indices = [feature_name_list.index("latitude"), feature_name_list.index("longitude")]
            latlon_to_draw_map = handcrafted_features[:, latlon_indices, :, :]  # (number_of_files, 2, N, T)
            draw_map = "latlon"

        if has_pose == True:
            skeleton_xy = post_analysis_dictionary["dictionary"]["skeleton_xy"][mask_class]  # (number_of_files, N, number_of_joints, T, (x, y))
            draw_interactive_map = "skeleton_xy"
        else:
            draw_interactive_map = "xy"

        # Create the class key
        if class_name_list[i] not in new_dictionary:
            new_dictionary[class_name_list[i]] = {}

        # save data
        number_of_files = np.sum(mask_class)
        for j in range(number_of_files):

            # Create the filename key
            filename = np.array(post_analysis_dictionary["dictionary"]["file_names"])[mask_class][j]
            if filename not in new_dictionary[class_name_list[i]]:
                new_dictionary[class_name_list[i]][filename] = {}

            # Get indices of not zero-padding
            valid_timestamps = np.any(
                handcrafted_features[j, :, :, :] != 0,
                axis=(0, 1),  # along hand-crafted features and number_of_nodes
            )  # (number_of_timestamps)
            non_zero_indices = np.where(valid_timestamps)[0]

            new_dictionary[class_name_list[i]][filename]["draw_map"] = draw_map
            new_dictionary[class_name_list[i]][filename]["draw_interactive_map"] = draw_interactive_map
            new_dictionary[class_name_list[i]][filename]["time_to_draw_map"] = time_to_draw_map[j, :, :, :][0, 0, non_zero_indices]  # (t)
            new_dictionary[class_name_list[i]][filename]["xy_to_draw_map"] = xy_to_draw_map[j, :, :, :][:, :, non_zero_indices]  # (2, N, t)
            new_dictionary[class_name_list[i]][filename]["latlon_to_draw_map"] = None if draw_map == "xy" else latlon_to_draw_map[j, :, :, :][:, 0, non_zero_indices]  # (2, t)
            new_dictionary[class_name_list[i]][filename]["skeleton_to_draw_map"] = None if draw_interactive_map == "xy" else skeleton_xy[j, :, :, :, :][:, :, non_zero_indices, :]  # (N, number_of_joints, t, (x, y))
            new_dictionary[class_name_list[i]][filename]["attentions"] = attentions[mask_class][j, :, :, :][:, :, non_zero_indices]
            new_dictionary[class_name_list[i]][filename]["handcrafted_features_unstandardized"] = handcrafted_features[j, :, :, :][:, :, non_zero_indices]  # (F, N, t)
            new_dictionary[class_name_list[i]][filename]["prediction"] = predictions_argmax[mask_class][j]  # (number_of_attention_branches)
            new_dictionary[class_name_list[i]][filename]["label"] = labels_argmax[mask_class][j]  # (number_of_attention_branches)
            new_dictionary[class_name_list[i]][filename]["distance_between_nodes"] = None if model_type == "single" else post_analysis_dictionary["dictionary"]["edge_data"][mask_class][j, 0, :, :][:, non_zero_indices]  # (E, t). 0: index of distanace between nodes

    with gzip.open("dictionary_for_visualization.pkl", "wb") as file:
        pickle.dump(new_dictionary, file)


def show_classification_report(
    dictionary,
    number_of_attention_branches,
):
    
    # Dropdown menu
    dropdown_left_0 = widgets.Dropdown(
        options=list(range(1, number_of_attention_branches + 1)),
        description="Attention",
    )

    # Output area
    output_left = widgets.Output()

    # Function to update the plot
    def update_left_plot(*args):

        with output_left:

            clear_output(wait=True)

            # Show table
            df = dictionary[("Attention" + str(dropdown_left_0.value))]
            df.rename(
                columns=lambda x: '' if isinstance(x, str) and x.startswith('Unnamed') else x,
                inplace=True
            )

            display(df)
            print("")
            print("Selected: Attention" + str(dropdown_left_0.value))

    # Monitoring for the dropdown menu
    dropdown_left_0.observe(update_left_plot, names="value")

    # Vertical arrangement of dropdown menus and output areas
    box_left = widgets.VBox([dropdown_left_0, output_left])

    # Display the dropdown menu and output area
    display(box_left)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization


def create_interactive_pca_plot(
    data,
    scatter_plot_text="Default",
    scale=2,

    # Figure (drawing area) size in pixels
    fig_width=900,
    fig_height=700,

    # 3D scene aspect
    aspectmode="cube",  # "cube", "data", "manual", "auto"

    # Font sizes
    axis_title_size=24,
    tick_size=17,
    point_text_size=24,

    # Marker / label tweaks
    marker_size=3,
    label_yshift=10,
):
    num_data, _ = data.shape

    # PCA (3 components)
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(data)

    x = pca_results[:, 0]
    y = pca_results[:, 1]
    z = pca_results[:, 2]

    # Labels
    if scatter_plot_text == "Default" or scatter_plot_text is None:
        texts = [f"Attention {i+1}" for i in range(num_data)]
    else:
        if len(scatter_plot_text) != num_data:
            raise ValueError(
                f"Length of scatter_plot_text ({len(scatter_plot_text)}) "
                f"does not match number of data points ({num_data})."
            )
        texts = list(scatter_plot_text)

    # Markers only (text is drawn as annotations to stay on top)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="magenta",
                    line=dict(width=1, color="magenta"),
                ),
                hovertext=texts,
                hoverinfo="text",
            )
        ]
    )

    # Overlay labels as scene annotations (top layer)
    annotations = [
        dict(
            x=float(xi),
            y=float(yi),
            z=float(zi),
            text=str(ti),
            showarrow=False,
            yshift=label_yshift,  # pixel shift upward
            font=dict(
                size=point_text_size,
                color="black",
            ),
            # Uncomment if you want a readable background for the text
            # bgcolor="rgba(255,255,255,0.7)",
            # bordercolor="black",
            # borderwidth=0,
        )
        for xi, yi, zi, ti in zip(x, y, z, texts)
    ]

    # Common axis styling
    axis_common = dict(
        linecolor="black",
        linewidth=5,
        backgroundcolor="white",
        showbackground=False,
        gridcolor="lightgrey",
        showticklabels=True,
        ticks="outside",
        tickwidth=2,
        tickcolor="black",
        ticklen=5,
        tickfont=dict(size=tick_size, color="black"),
    )

    fig.update_layout(
        # Set drawing area size (pixels)
        width=fig_width,
        height=fig_height,

        # title=dict(
        #     text="PCA 3D Visualization",
        #     font=dict(size=24, color="black"),
        # ),
        font=dict(color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            annotations=annotations,

            # Set 3D scene aspect ratio behavior
            aspectmode=aspectmode,

            xaxis=dict(
                title=dict(
                    text="PC1",
                    font=dict(size=axis_title_size, color="black"),
                ),
                **axis_common,
            ),
            yaxis=dict(
                title=dict(
                    text="PC2",
                    font=dict(size=axis_title_size, color="black"),
                ),
                **axis_common,
            ),
            zaxis=dict(
                title=dict(
                    text="PC3",
                    font=dict(size=axis_title_size, color="black"),
                ),
                **axis_common,
            ),
        ),
        showlegend=False,
    )

    fig.show(
        config={
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "pca_3d_visualization",
                "scale": scale,
            }
        }
    )


def plot_distribution_of_handcrafted_feature(
    dictionary,
    class_name_list,
    comparison_name,
    feature_name,
    dpi,
):

    match = re.search(r'Condition (\d+) - (\d+)', comparison_name)
    i, j = (int(match.group(1)) - 1), (int(match.group(2)) - 1)

    plt.figure(figsize=(9, 3), dpi=dpi)
    plt.hist(dictionary[i], bins=50, alpha=0.5, color='tab:blue', label=class_name_list[i])
    plt.hist(dictionary[j], bins=50, alpha=0.5, color='tab:orange', label=class_name_list[j])

    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


def show_distribution_of_handcrafted_features_and_gui(
    dictionary,
    class_name_list,
    feature_name_list,
    dpi,
):
    
    # Dropdown for comparison selection
    comparison_name_list = list(dictionary.keys())
    comparison_dropdown = widgets.Dropdown(
        options=comparison_name_list,
        description="Comparison",
    )

    # Dropdown for feature selection
    feature_dropdown = widgets.Dropdown(
        options=feature_name_list,
        description="Feature",
    )

    # Output area
    output_left = widgets.Output()

    # Function to update plot
    def update_left_plot(*args):
        with output_left:
            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", comparison_dropdown.value)
            print("    ⭐️", feature_dropdown.value)
            print("")

            plot_distribution_of_handcrafted_feature(
                dictionary[comparison_dropdown.value][feature_dropdown.value],
                class_name_list,
                comparison_dropdown.value,
                feature_dropdown.value,
                dpi,
            )

    # Monitoring for the dropdown menu
    comparison_dropdown.observe(update_left_plot, names="value")
    feature_dropdown.observe(update_left_plot, names="value")

    # Arrange widgets and output area
    controls_box = widgets.VBox([comparison_dropdown, feature_dropdown])
    display_box = widgets.VBox([controls_box, output_left])

    # Display the menus and output areas
    display(display_box)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization


def plot_single_trajectory_with_marker_and_attention(
    draw_map,
    coords_to_draw_map,
    time_to_draw_map,
    attention_weight,
):
    
    # Exclude all indices containing NaNs
    mask = (
        ~np.isnan(coords_to_draw_map[0, :]) &
        ~np.isnan(coords_to_draw_map[1, :])
    )

    coords_to_draw_map = coords_to_draw_map[:, mask].astype(float)  # Avoid an error "TypeError: unsupported type for timedelta seconds component: numpy.float32"
    time_to_draw_map   = time_to_draw_map[mask].astype(float)
    attention_weight   = attention_weight[mask].astype(float)

    if draw_map == "xy":
        # Dummy range of latitude and longitude
        LAT_RANGE = [35, 36]  # Range of latitude
        LON_MIN = 139  # Minimum longitude

        # Conversion ratio
        ratio = (LAT_RANGE[1] - LAT_RANGE[0]) / (np.max(coords_to_draw_map[0, :]) - np.min(coords_to_draw_map[0, :]))

        # Convert x, y coordinates to latitude and longitude
        lon = ((coords_to_draw_map[0] - np.min(coords_to_draw_map[0])) * ratio + LAT_RANGE[0])
        lat = ((coords_to_draw_map[1] - np.min(coords_to_draw_map[1])) * ratio + LAT_RANGE[1])
    if draw_map == "latlon":
        lat = coords_to_draw_map[0, :]
        lon = coords_to_draw_map[1, :]

    # Calculate the center of the map
    center_lat = np.mean(lat)
    center_lon = np.mean(lon)

    # Create map object
    if draw_map == "xy":
        map = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles=None)
    if draw_map == "latlon":
        map = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Normalize the strength of attention to range from 0 to 1
    normalized_attention_weight = (attention_weight - np.min(attention_weight)) / (np.max(attention_weight) - np.min(attention_weight))

    # Create a color map (linear gradient from black to magenta)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", "magenta"])

    # Draw each segment of the trajectory
    for i in range(len(lat) - 1):
        latlon1 = [lat[i], lon[i]]
        latlon2 = [lat[i + 1], lon[i + 1]]

        # Determine color using the color map
        color = cmap(normalized_attention_weight[i])

        # Convert RGB color to hex
        color_hex = matplotlib.colors.to_hex(color)

        # Draw segment
        folium.PolyLine(locations=[latlon1, latlon2], color=color_hex, weight=2.5).add_to(map)

    # Set the base time
    base_time = datetime.datetime(2000, 1, 1, 0, 0, 0)

    # Create GeoJSON format data
    features = []
    for i in range(len(lat) - 1):
        timestamp = base_time + datetime.timedelta(seconds=time_to_draw_map[i])
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon[i], lat[i]],
            },
            "properties": {
                "time": timestamp.strftime("%Y-%m-%dT%H:%M:%S"),
                "style": {"color": ""},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "blue",
                    "fillOpacity": 0.6,
                    "stroke": "false",
                    "radius": 7
                }
            }
        }
        features.append(feature)

    # Add TimestampedGeoJson
    TimestampedGeoJson({
        "type": "FeatureCollection",
        "features": features,
    }, period="PT1S",
    duration="PT1S",
    add_last_point=False,
    auto_play=False,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
    ).add_to(map)

    map.fit_bounds([[np.min(lat), np.min(lon)], [np.max(lat), np.max(lon)]])

    # Display the map as HTML
    map_html = map._repr_html_()
    # Set iframe size smaller
    iframe_style = '<div style="width: 600px; height: 600px;">{}</div>'.format(map_html)

    return display(HTML(iframe_style))


def plot_trajectory_with_attention_and_handcrafted_feature(
    model_type,
    xy_to_draw_map,
    attention_weight,
    handcrafted_feature,
    feature_name,
    node_to_plot,
    dpi,
    xlim,
    ylim,
):

    if model_type == "single":

        with tqdm(
            initial=0,
            total=4, 
            desc="Building plot", 
            unit="step", 
            bar_format="{desc}: {percentage:3.0f}% |{bar}|"
        ) as pbar:

            # Exclude all indices containing NaNs
            mask = (
                ~np.isnan(xy_to_draw_map[0, :]) &
                ~np.isnan(xy_to_draw_map[1, :])
            )

            xy_to_draw_map      = xy_to_draw_map[:, mask]
            attention_weight    = attention_weight[mask]
            handcrafted_feature = handcrafted_feature[mask]

            # Plot the trajectory with attention highlight
            t = len(attention_weight)
            attention_threshold = 1 / t

            # Create figure and axes
            fig, ax = plt.subplots(2, 1, figsize=(6, 6), dpi=dpi)

            # Plot the trajectory and color each segment based on the strength of attention
            x = xy_to_draw_map[0, :]
            y = xy_to_draw_map[1, :]

            # Plot the entire trajectory using a light color
            ax[0].plot(x, y, color="lightgray", lw=2, label=f"Trajectory")
            
            # Highlight the attention period using magenta color
            for j in range(t - 1):
                if attention_weight[j] > attention_threshold:
                    ax[0].plot(x[j:j+2], y[j:j+2], color="magenta", lw=2)

            pbar.update(1)

            # Legend with a dummy lined colored by magenta 
            highlight_line = mlines.Line2D([], [], color="magenta", lw=2, label="Highlighted by attention")
            handles, labels = ax[0].get_legend_handles_labels()
            handles.append(highlight_line)
            ax[0].legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            ax[0].set_title("Trajectory with attention", fontsize=14)
            ax[0].set_xlabel("x", fontsize=12)
            ax[0].set_ylabel("y", fontsize=12)
            if xlim:
                ax[0].set_xlim(xlim)
            if ylim:
                ax[0].set_ylim(ylim)
            ax[0].set_aspect("equal", adjustable="box")
            ax[0].grid(True)

            pbar.update(1)

            # Plot the handcrafted feature and the attention with reversed axes
            ax[1].plot(list(range(len(handcrafted_feature))), handcrafted_feature, color="black")
            ax[1].set_xlabel("Time", fontsize=12)
            ax[1].set_ylabel(feature_name, color="black", fontsize=12)
            ax[1].tick_params(axis="y", labelcolor="black")

            # Right axis and plot for the first subplot
            ax0_right = ax[1].twinx()
            ax0_right.plot(list(range(len(handcrafted_feature))), attention_weight, color="magenta")
            ax0_right.set_ylabel("Attention", color="magenta", fontsize=12)
            ax0_right.tick_params(axis="y", labelcolor="magenta")
            ax[1].set_title("Handcrafted-feature and attention", fontsize=14)

            pbar.update(1)

            plt.tight_layout()
            plt.show()

            pbar.update(1)

            plt.cla()
            plt.clf()
            plt.close()


    if model_type == "multi":

        with tqdm(
            initial=0,
            total=5, 
            desc="Building plot", 
            unit="step", 
            bar_format="{desc}: {percentage:3.0f}% |{bar}|"
        ) as pbar:

            number_of_nodes, t = attention_weight.shape

            # Adjust the figure size
            fig, ax = plt.subplots(3, 1, figsize=(6, 12), dpi=dpi)

            # Plot the trajectory with attention highlight
            attention_threshold = 1 / (t * number_of_nodes)

            # Retrieve the colormap
            colors = plt.get_cmap("viridis")
            color_array = colors(np.linspace(0, 1, number_of_nodes))

            # Plot the trajectory for each node
            for i in range(number_of_nodes):
                x = xy_to_draw_map[0, i, :]
                y = xy_to_draw_map[1, i, :]

                # Define the base color (make it lighter with alpha=0.5)
                base_color = color_array[i]
                light_color = base_color.copy()
                light_color[3] = 0.5  # Set the alpha value to 0.5 for transparency

                # Plot the entire trajectory using a light color
                ax[0].plot(x, y, color=light_color, lw=1, label=f"{i+1}")
                
                # Add text label at the start of each node's trajectory 
                ax[0].text(x[0], y[0], str(i + 1), fontsize=10, fontweight="bold", ha="center", va="bottom") 

                # Highlight the attention period using magenta color
                for j in range(t - 1):
                    if attention_weight[i, j] > attention_threshold:
                        ax[0].plot(x[j:j+2], y[j:j+2], color="magenta", lw=2)

            pbar.update(1)

            # Legend with a dummy lined colored by magenta 
            highlight_line = mlines.Line2D([], [], color="magenta", lw=2, label="Highlighted by attention")
            handles, labels = ax[0].get_legend_handles_labels()
            handles.append(highlight_line)
            ax[0].legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

            # Set equal scaling for both axes
            ax[0].set_aspect("equal", adjustable="box")

            # Add title, axis labels, and legend
            ax[0].set_title("Trajectory with attention", fontsize=16)
            ax[0].set_xlabel("x", fontsize=14)
            ax[0].set_ylabel("y", fontsize=14)
            if xlim:
                ax[0].set_xlim(xlim)
            if ylim:
                ax[0].set_ylim(ylim)
            ax[0].grid(True)

            pbar.update(1)

            # Plot the heatmap of the attention
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", "magenta"])
            attention_vmin = attention_weight.min()
            attention_vmax = attention_weight.max()
            if attention_vmin == attention_vmax:
                attention_vmax += 1e-5
            ax[1].imshow(attention_weight, aspect="auto", cmap=cmap, vmin=attention_vmin, vmax=attention_vmax)
            ax[1].set_xlabel("Time", fontsize=14)
            ax[1].set_ylabel("Node", fontsize=14)
            ax[1].set_yticks(np.arange(number_of_nodes))
            ax[1].set_yticklabels(np.arange(1, number_of_nodes + 1))
            ax[1].set_title("Attention value", fontsize=16)
            fig.colorbar(ax[1].images[0], ax=ax[1])

            pbar.update(1)

            # Plot the handcrafted feature and the attention with reversed axes
            # Left axis
            ax[2].plot(list(range(len(handcrafted_feature[node_to_plot, :]))), handcrafted_feature[node_to_plot, :], color="black")
            ax[2].set_xlabel("Time", fontsize=14)
            ax[2].set_ylabel(feature_name, color="black", fontsize=14)
            ax[2].tick_params(axis="y", labelcolor="black")

            # Right axis
            ax2_right = ax[2].twinx()
            ax2_right.plot(list(range(len(handcrafted_feature[node_to_plot, :]))), attention_weight[node_to_plot, :], color="magenta")
            ax2_right.set_ylabel("Attention", color="magenta", fontsize=14)
            ax2_right.tick_params(axis="y", labelcolor="magenta")
            ax2_right.set_ylim(attention_vmin, attention_vmax)  # Set y-axis limits to match the attention value range
            ax[2].set_title("Handcrafted-feature and attention", fontsize=16)

            pbar.update(1)

            plt.tight_layout()
            plt.show()

            pbar.update(1)

            plt.cla()
            plt.clf()
            plt.close()

# Initialize global variables to store dropdown values
global_selected_values = {
    "left_0": None, "left_1": None, "left_2": None,
    "right_0": None, "right_1": None, "right_2": None,
}


def make_plots_and_gui(
    dictionary,
    model_type,
    class_name_list,
    default_filename_list,
    number_of_attention_branches,
    feature_name_list,
    number_of_nodes,
    dpi,
    xlim,
    ylim,
):

    # Menus on the left side
    dropdown_left_0 = widgets.Dropdown(
        options=class_name_list,
        description="Condition",
    )
    dropdown_left_1 = widgets.Dropdown(
        options=default_filename_list,
        description="Filename",
    )
    dropdown_left_2 = widgets.Dropdown(
        options=list(range(1, number_of_attention_branches + 1)),
        description="Attention",
    )
    dropdown_left_3 = widgets.Dropdown(
        options=feature_name_list,
        description="Feature",
    )

    if model_type == "multi":
        dropdown_left_4 = widgets.Dropdown(
            options=list(range(1, number_of_nodes + 1)),
            description="Node",
        )
    if model_type == "single":
        dropdown_left_4 = widgets.Dropdown(
            options=list(range(1, number_of_nodes + 1)),
            description="Node",
            layout=widgets.Layout(display="none"),
        )

    # Menus on the right side
    dropdown_right_0 = widgets.Dropdown(
        options=class_name_list,
        description="Condition",
    )
    dropdown_right_1 = widgets.Dropdown(
        options=default_filename_list,
        description="Filename",
    )
    dropdown_right_2 = widgets.Dropdown(
        options=list(range(1, number_of_attention_branches + 1)),
        description="Attention",
    )
    dropdown_right_3 = widgets.Dropdown(
        options=feature_name_list,
        description="Feature",
    )

    if model_type == "multi":
        dropdown_right_4 = widgets.Dropdown(
            options=list(range(1, number_of_nodes + 1)),
            description="Node",
        )
    if model_type == "single":
        dropdown_right_4 = widgets.Dropdown(
            options=list(range(1, number_of_nodes + 1)),
            description="Node",
            layout=widgets.Layout(display="none"),
        )

    # Output areas
    output_left = widgets.Output()
    output_right = widgets.Output()

    # Function to update the dropdown list of filenames
    def update_left_filenames(*args):
        dropdown_left_1.options = dictionary[dropdown_left_0.value].keys()

    def update_right_filenames(*args):
        dropdown_right_1.options = dictionary[dropdown_right_0.value].keys()

    # Function to update the plot on the left side
    def update_left_plot(*args):

        with output_left:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_left_0.value)
            print("    ⭐️", dropdown_left_1.value)
            print("    ⭐️ Attention" + str(dropdown_left_2.value))
            print("    ⭐️", dropdown_left_3.value)
            if model_type == "multi":
                print("    ⭐️ Node" + str(dropdown_left_4.value))

            prediction = dictionary[dropdown_left_0.value][dropdown_left_1.value]["prediction"][(int(dropdown_left_2.value) - 1)]
            label = dictionary[dropdown_left_0.value][dropdown_left_1.value]["label"][(int(dropdown_left_2.value) - 1)]

            print("")
            if prediction == label:
                print("🟩 Model prediction is correct 🟩")
                print("    Prediction:", class_name_list[prediction])
            else:
                print("⬜️ Model prediction is not correct ⬜️")
                print("    Prediction:", class_name_list[prediction])
            print("")

            plot_trajectory_with_attention_and_handcrafted_feature(
                model_type,
                dictionary[dropdown_left_0.value][dropdown_left_1.value]["xy_to_draw_map"],
                dictionary[dropdown_left_0.value][dropdown_left_1.value]["attentions"][(int(dropdown_left_2.value) - 1), :],
                dictionary[dropdown_left_0.value][dropdown_left_1.value]["handcrafted_features_unstandardized"][feature_name_list.index(dropdown_left_3.value), :],
                dropdown_left_3.value,
                (dropdown_left_4.value - 1),
                dpi,
                xlim,
                ylim,
            )

            # Update global variables whenever a dropdown value changes
            global global_selected_values  # Declare as global to modify it
            global_selected_values["left_0"] = dropdown_left_0.value
            global_selected_values["left_1"] = dropdown_left_1.value
            global_selected_values["left_2"] = dropdown_left_2.value

    # Function to update the plot on the right side
    def update_right_plot(*args):

        with output_right:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_right_0.value)
            print("    ⭐️", dropdown_right_1.value)
            print("    ⭐️ Attention" + str(dropdown_right_2.value))
            print("    ⭐️", dropdown_right_3.value)
            if model_type == "multi":
                print("    ⭐️ Node" + str(dropdown_right_4.value))

            prediction = dictionary[dropdown_right_0.value][dropdown_right_1.value]["prediction"][(int(dropdown_right_2.value) - 1)]
            label = dictionary[dropdown_right_0.value][dropdown_right_1.value]["label"][(int(dropdown_right_2.value) - 1)]

            print("")
            if prediction == label:
                print("🟩 Model prediction is correct 🟩")
                print("    Prediction:", class_name_list[prediction])
            else:
                print("⬜️ Model prediction is not correct ⬜️")
                print("    Prediction:", class_name_list[prediction])
            print("")

            plot_trajectory_with_attention_and_handcrafted_feature(
                model_type,
                dictionary[dropdown_right_0.value][dropdown_right_1.value]["xy_to_draw_map"],
                dictionary[dropdown_right_0.value][dropdown_right_1.value]["attentions"][(int(dropdown_right_2.value) - 1), :],
                dictionary[dropdown_right_0.value][dropdown_right_1.value]["handcrafted_features_unstandardized"][feature_name_list.index(dropdown_right_3.value), :],
                dropdown_right_3.value,
                (dropdown_right_4.value - 1),
                dpi,
                xlim,
                ylim,
            )

            # Update global variables whenever a dropdown value changes
            global global_selected_values  # Declare as global to modify it
            global_selected_values["right_0"] = dropdown_right_0.value
            global_selected_values["right_1"] = dropdown_right_1.value
            global_selected_values["right_2"] = dropdown_right_2.value

    # Monitoring for the left side menus
    dropdown_left_0.observe(update_left_filenames, names="value")
    dropdown_left_1.observe(update_left_plot, names="value")
    dropdown_left_2.observe(update_left_plot, names="value")
    dropdown_left_3.observe(update_left_plot, names="value")
    dropdown_left_4.observe(update_left_plot, names="value")

    # Monitoring for the right side dropdown menus
    dropdown_right_0.observe(update_right_filenames, names="value")
    dropdown_right_1.observe(update_right_plot, names="value")
    dropdown_right_2.observe(update_right_plot, names="value")
    dropdown_right_3.observe(update_right_plot, names="value")
    dropdown_right_4.observe(update_right_plot, names="value")

    # Vertical arrangement of menus and output areas
    box_left = widgets.VBox([dropdown_left_0, dropdown_left_1, dropdown_left_2, dropdown_left_3, dropdown_left_4, output_left])
    box_right = widgets.VBox([dropdown_right_0, dropdown_right_1, dropdown_right_2, dropdown_right_3, dropdown_right_4, output_right])
    display_box = widgets.HBox([box_left, box_right])

    # Display the menus and output areas
    display(display_box)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization
    update_right_plot(None)  # Call the function once for initialization


def plot_multi_trajectory_with_marker_and_attention(
    coords_to_draw_map,
    attention_weight,
    sampling_interval,
):

        number_of_nodes, t = attention_weight.shape

        # Parameter settings
        sampled_steps = np.arange(0, t, sampling_interval)
        
        # Use sampled data
        sampled_data = coords_to_draw_map[:, :, sampled_steps]
        attention_weight = attention_weight[:, sampled_steps]

        # Normalize the attention weights to range from 0 to 1
        normalized_attention_weight = (attention_weight - np.min(attention_weight)) / (np.max(attention_weight) - np.min(attention_weight))

        # Calculate mean attention over N nodes at each timestep
        mean_attention = attention_weight.mean(axis=0)  # shape: (len(sampled_steps),)

        # Normalize mean attention to 0-1 for coloring
        normalized_mean_attention = (mean_attention - np.min(mean_attention)) / (np.max(mean_attention) - np.min(mean_attention))

        # Create a colormap from white or light gray to magenta
        cmap_marker = mcolors.LinearSegmentedColormap.from_list("", ["white", "magenta"])
        cmap_lineplot = mcolors.LinearSegmentedColormap.from_list("", ["lightgray", "magenta"])

        # Obtain colormap object for node paths
        colors = plt.colormaps['viridis']
        color_array = colors(np.linspace(0, 1, number_of_nodes))

        # Convert RGBA values to Plotly-compatible RGBA strings
        def rgba_to_plotly_str(rgba_tuple):
            r, g, b, a = rgba_tuple
            return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})'

        # Set opacity to 0.5 for node paths
        color_list_plotly = [rgba_to_plotly_str((*color[:3], 0.5)) for color in color_array]

        # Colour every marker in one call rather than once per node and frame.
        marker_rgba = cmap_marker(normalized_attention_weight)  # (number_of_nodes, frames, 4)
        marker_colors = [
            [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})' for r, g, b, a in node_rgba]
            for node_rgba in marker_rgba
        ]

        # Set initial state for each vertex
        frames = []
        initial_data = []

        # First, add paths
        for i in range(number_of_nodes):
            color = color_list_plotly[i]
            initial_data.append(dict(
                type='scatter',
                x=[sampled_data[0, i, 0]],
                y=[sampled_data[1, i, 0]],
                mode='lines',
                name=f'{i+1} trajectory',
                line=dict(color=color),
                opacity=0.9,  # Make paths semi-transparent
                legendrank=1  # Set legend order for paths
            ))

        # Next, add markers
        for i in range(number_of_nodes):
            marker_color = marker_colors[i][0]
            initial_data.append(dict(
                type='scatter',
                x=[sampled_data[0, i, 0]],
                y=[sampled_data[1, i, 0]],
                mode='markers+text',  # Display markers and text
                marker=dict(size=20, color=marker_color),
                text=str(i + 1),  # Display vertex number inside marker
                textfont=dict(color='black'),
                textposition="middle center",  # Position text in center
                name=f'{i+1} marker',
                opacity=1,  # Do not make markers transparent
                legendrank=0  # Set legend order for markers (display before paths)
            ))

        # Add data to each frame
        for t in tqdm(range(1, sampled_data.shape[2]), desc="Building frames"): 
            frame_data = []
            # First, add paths
            for i in range(number_of_nodes):
                color = color_list_plotly[i]
                frame_data.append(dict(
                    type='scatter',
                    x=sampled_data[0, i, :t+1],
                    y=sampled_data[1, i, :t+1],
                    mode='lines',
                    line=dict(color=color),
                    opacity=0.9,
                ))
            # Next, add markers
            for i in range(number_of_nodes):
                marker_color = marker_colors[i][t]
                frame_data.append(dict(
                    type='scatter',
                    x=[sampled_data[0, i, t]],
                    y=[sampled_data[1, i, t]],
                    mode='markers+text',
                    marker=dict(size=20, color=marker_color),
                    text=str(i + 1),
                    textfont=dict(color='black'),
                    textposition="middle center",
                    opacity=1,
                ))
            frames.append(dict(data=frame_data, name=str(t)))

        # Slider settings
        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Timestep: "},
            pad={"t": 20},
            steps=[dict(
                method="animate",
                args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                label=str(k * sampling_interval)
            ) for k in range(sampled_data.shape[2])]
        )]

        with tqdm(
            initial=0,
            total=11, 
            desc="Building plot", 
            unit="step", 
            bar_format="{desc}: {percentage:3.0f}% |{bar}|"
        ) as pbar:
        
            # Calculate axis ranges based on data
            x_min = sampled_data[0, :, :].min()
            x_max = sampled_data[0, :, :].max()
            y_min = sampled_data[1, :, :].min()
            y_max = sampled_data[1, :, :].max()

            # Prepare data for the mean attention line graph
            x_line = sampled_steps
            y_line = mean_attention

            pbar.update(1)

            # Create line colors based on normalized mean attention
            # Convert normalized values to colors using the colormap
            line_colors_rgba = [cmap_lineplot(value) for value in normalized_mean_attention]
            # Convert RGBA values to hex strings
            def rgba_to_hex(rgba_tuple):
                r, g, b, a = rgba_tuple
                return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

            line_colors_hex = [rgba_to_hex(color) for color in line_colors_rgba]

            pbar.update(1)

            # Create line segments with individual colors
            line_segments = []
            for i in range(len(x_line) - 1):
                segment = dict(
                    type='scatter',
                    x=x_line[i:i+2],
                    y=y_line[i:i+2],
                    mode='lines',
                    line=dict(color=line_colors_hex[i], width=2),
                    showlegend=False,
                    hoverinfo='skip'
                )
                line_segments.append(segment)
                
            pbar.update(1)

            # Create subplots
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.02,
                row_heights=[0.8, 0.1]
            )

            pbar.update(1)

            # Add trajectory data to the first subplot
            for trace in initial_data:
                fig.add_trace(trace, row=1, col=1)

            pbar.update(1)

            # Add the line segments to the second subplot
            for segment in line_segments:
                fig.add_trace(segment, row=2, col=1)

            pbar.update(1)

            # Update layout
            fig.update_layout(
                title="Trajectory",
                sliders=sliders,
                updatemenus=[dict(
                    type="buttons",
                    buttons=[dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]
                    )]
                )],
                height=500  # Adjust the height as needed
            )

            pbar.update(1)

            # Update axes for the trajectory plot
            fig.update_xaxes(
                range=[x_min, x_max],
                showticklabels=False,
                row=1,
                col=1
            )
            fig.update_yaxes(
                range=[y_min, y_max],
                scaleanchor="x",
                scaleratio=1,
                showticklabels=False,
                row=1,
                col=1
            )

            pbar.update(1)

            # Update axes for the mean attention line graph
            fig.update_xaxes(
                title_text="Time",
                tickformat=',',  # Ensure numbers are displayed without 'k'
                row=2,
                col=1
            )
            fig.update_yaxes(
                title_text="Mean attention",
                visible=False,  # Hide y-axis labels to make it narrow
                row=2,
                col=1
            )

            pbar.update(1)

            # Assign frames to the figure
            fig.frames = frames

            pbar.update(1)

            # Display plot
            display(fig)

            pbar.update(1)


def plot_trajectory_with_marker_and_attention(
    model_type,
    draw_map,
    draw_interactive_map,
    time_to_draw_map,
    xy_to_draw_map,
    latlon_to_draw_map,
    skeleton_to_draw_map,
    attention_weight,
    sampling_interval,
):

    if (model_type == "multi") or (model_type == "single" and draw_interactive_map == "skeleton_xy"):
        if model_type == "multi":
            coords_to_draw_map = xy_to_draw_map
        if model_type == "single" and draw_interactive_map == "skeleton_xy":
            coords_to_draw_map = skeleton_to_draw_map.transpose(3, 1, 2, 0).squeeze(3)

        plot_multi_trajectory_with_marker_and_attention(
            coords_to_draw_map,
            attention_weight,
            sampling_interval,
        )

    else:
        if draw_map == "xy":
            coords_to_draw_map = xy_to_draw_map[:, 0, :]
        if draw_map == "latlon":
            coords_to_draw_map = latlon_to_draw_map[:, 0, :]

        plot_single_trajectory_with_marker_and_attention(
            draw_map,
            coords_to_draw_map,
            time_to_draw_map,
            attention_weight.squeeze(0),
        )


def make_interactive_plot_and_gui(
    dictionary,
    model_type,
    sampling_interval,
):

    # Menu
    dropdown = widgets.Dropdown(
        options=["Left plot", "Right plot"],
    )

    # Button to create the plot
    create_plot_button = widgets.Button(
        description="Create plot"
    )

    # Output area
    output = widgets.Output()

    # Function to update the plot
    def update_plot(*args):

        with output:

            clear_output(wait=True)

            global global_selected_values
            dropdown_left_0 = global_selected_values["left_0"]
            dropdown_left_1 = global_selected_values["left_1"]
            dropdown_left_2 = global_selected_values["left_2"]
            dropdown_right_0 = global_selected_values["right_0"]
            dropdown_right_1 = global_selected_values["right_1"]
            dropdown_right_2 = global_selected_values["right_2"]

            if dropdown.value == "Left plot":

                print("\nSelected:")
                print(dropdown.value)
                print("    ⭐️", dropdown_left_0)
                print("    ⭐️", dropdown_left_1)
                print("    ⭐️ Attention" + str(dropdown_left_2))
                print("")

                plot_trajectory_with_marker_and_attention(
                    model_type,
                    dictionary[dropdown_left_0][dropdown_left_1]["draw_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["draw_interactive_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["time_to_draw_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["xy_to_draw_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["latlon_to_draw_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["skeleton_to_draw_map"],
                    dictionary[dropdown_left_0][dropdown_left_1]["attentions"][(int(dropdown_left_2) - 1), :],
                    sampling_interval,
                )

            elif dropdown.value == "Right plot":

                print("\nSelected:")
                print(dropdown.value)
                print("    ⭐️", dropdown_right_0)
                print("    ⭐️", dropdown_right_1)
                print("    ⭐️ Attention" + str(dropdown_right_2))
                print("")

                plot_trajectory_with_marker_and_attention(
                    model_type,
                    dictionary[dropdown_right_0][dropdown_right_1]["draw_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["draw_interactive_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["time_to_draw_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["xy_to_draw_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["latlon_to_draw_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["skeleton_to_draw_map"],
                    dictionary[dropdown_right_0][dropdown_right_1]["attentions"][(int(dropdown_right_2) - 1), :],
                    sampling_interval,
                )

    # Call update_plot when the button is clicked
    create_plot_button.on_click(update_plot)

    # Vertical arrangement of menus, button, and output areas
    display_box = widgets.VBox([dropdown, create_plot_button, output])

    # Display the menus, button, and output area
    display(display_box)

    # Do not display the plot on initialization


def color_cells(
    value,
):
    """
    Show colored table.
    """

    if not isinstance(value, (int, float)):
        return "background-color: transparent"

    if value > 0.3:
        color = "#FF9999"
    elif value > 0.2:
        color = "#E6A8A8"
    elif value > 0.1:
        color = "#CCB8B8"
    elif value < -0.3:
        color = "#80E6E6"
    elif value < -0.2:
        color = "#99D6D6"
    elif value < -0.1:
        color = "#B3C7C7"
    else:
        return "background-color: transparent"
    
    return f"background-color: {color}; color: black"


def show_correlation_table_and_gui(
    dictionary,
    class_name_list,
    number_of_attention_branches,
):

    # Dropdown menus on the left side
    dropdown_left_0 = widgets.Dropdown(
        options=class_name_list, 
        description="Condition",
    )
    dropdown_left_1 = widgets.Dropdown(
        options=list(range(1, number_of_attention_branches + 1)),
        description="Attention",
    )

    # Dropdown menus on the right side
    dropdown_right_0 = widgets.Dropdown(
        options=class_name_list, 
        description="Condition",
    )
    dropdown_right_1 = widgets.Dropdown(
        options=list(range(1, number_of_attention_branches + 1)),
        description="Attention",
    )

    # Output areas
    output_left = widgets.Output()
    output_right = widgets.Output()

    # Function to update the plot on the left side
    def update_left_plot(*args):

        with output_left:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_left_0.value)
            print("    ⭐️ Attention" + str(dropdown_left_1.value)) 

            # Show colored table
            df = dictionary[dropdown_left_0.value][["Feature", ("Attention" + str(dropdown_left_1.value))]]
            df = df.sort_values(by=("Attention" + str(dropdown_left_1.value)), ascending=False)

            # Reset the index to add a new sequential index
            df = df.reset_index(drop=True)
            df.index += 1  # Start index from 1

            # Apply the styling
            df = df.style.map(color_cells)

            display(df)

    # Function to update the plot on the right side
    def update_right_plot(*args):

        with output_right:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_right_0.value)
            print("    ⭐️ Attention" + str(dropdown_right_1.value)) 

            # Show colored table
            df = dictionary[dropdown_right_0.value][["Feature", ("Attention" + str(dropdown_right_1.value))]]
            df = df.sort_values(by=("Attention" + str(dropdown_right_1.value)), ascending=False)

            # Reset the index to add a new sequential index
            df = df.reset_index(drop=True)
            df.index += 1  # Start index from 1

            # Apply the styling
            df = df.style.map(color_cells)

            display(df)

    # Monitoring for the left side dropdown menus
    dropdown_left_0.observe(update_left_plot, names="value")
    dropdown_left_1.observe(update_left_plot, names="value")

    # Monitoring for the right side dropdown menus
    dropdown_right_0.observe(update_right_plot, names="value")
    dropdown_right_1.observe(update_right_plot, names="value")

    # Vertical arrangement of dropdown menus and output areas
    box_left = widgets.VBox([dropdown_left_0, dropdown_left_1, output_left])
    box_right = widgets.VBox([dropdown_right_0, dropdown_right_1, output_right])
    display_box = widgets.HBox([box_left, box_right])

    # Display the dropdown menus and output areas
    display(display_box)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization
    update_right_plot(None)  # Call the function once for initialization


def show_hellinger_distance_table_and_gui(
    dictionary,
    comparison_name_list,
):

    # Dropdown menus on the left side
    dropdown_left_0 = widgets.Dropdown(
        options=comparison_name_list, 
        description="Comparison",
    )

    # Dropdown menus on the right side
    dropdown_right_0 = widgets.Dropdown(
        options=comparison_name_list, 
        description="Comparison",
    )

    # Output areas
    output_left = widgets.Output()
    output_right = widgets.Output()

    # Function to update the plot on the left side
    def update_left_plot(*args):

        with output_left:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_left_0.value)

            # Show colored table
            df = dictionary["Sheet1"][["Feature", str(dropdown_left_0.value)]]
            df = df.sort_values(by=str(dropdown_left_0.value), ascending=False)

            # Reset the index to add a new sequential index
            df = df.reset_index(drop=True)
            df.index += 1  # Start index from 1

            # Apply the styling
            df = df.style.map(color_cells)

            display(df)

    # Function to update the plot on the right side
    def update_right_plot(*args):

        with output_right:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_right_0.value)

            # Show colored table
            df = dictionary["Sheet1"][["Feature", str(dropdown_right_0.value)]]
            df = df.sort_values(by=str(dropdown_right_0.value), ascending=False)

            # Reset the index to add a new sequential index
            df = df.reset_index(drop=True)
            df.index += 1  # Start index from 1

            # Apply the styling
            df = df.style.map(color_cells)

            display(df)

    # Monitoring for the left side dropdown menus
    dropdown_left_0.observe(update_left_plot, names="value")

    # Monitoring for the right side dropdown menus
    dropdown_right_0.observe(update_right_plot, names="value")

    # Vertical arrangement of dropdown menus and output areas
    box_left = widgets.VBox([dropdown_left_0, output_left])
    box_right = widgets.VBox([dropdown_right_0, output_right])
    display_box = widgets.HBox([box_left, box_right])

    # Display the dropdown menus and output areas
    display(display_box)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization
    update_right_plot(None)  # Call the function once for initialization


def show_table_and_gui(
    dictionary,
    dropdown_list,
    dropdown_name,
):

    # Dropdown menus on the left side
    dropdown_left_0 = widgets.Dropdown(
        options=dropdown_list, 
        description=dropdown_name,
    )

    # Dropdown menus on the right side
    dropdown_right_0 = widgets.Dropdown(
        options=dropdown_list, 
        description=dropdown_name,
    )

    # Output areas
    output_left = widgets.Output()
    output_right = widgets.Output()

    # Function to update the plot on the left side
    def update_left_plot(*args):

        with output_left:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_left_0.value)

            # Show the table
            df = dictionary[dropdown_left_0.value]
            display(df)

    # Function to update the plot on the right side
    def update_right_plot(*args):

        with output_right:

            clear_output(wait=True)

            print("\nSelected:")
            print("    ⭐️", dropdown_right_0.value)

            # Show the table
            df = dictionary[dropdown_right_0.value]
            display(df)

    # Monitoring for the left side dropdown menus
    dropdown_left_0.observe(update_left_plot, names="value")

    # Monitoring for the right side dropdown menus
    dropdown_right_0.observe(update_right_plot, names="value")

    # Vertical arrangement of dropdown menus and output areas
    box_left = widgets.VBox([dropdown_left_0, output_left])
    box_right = widgets.VBox([dropdown_right_0, output_right])
    display_box = widgets.HBox([box_left, box_right])

    # Display the dropdown menus and output areas
    display(display_box)

    # Display the initial graph
    update_left_plot(None)  # Call the function once for initialization
    update_right_plot(None)  # Call the function once for initialization


# ---------------------------------------------------------------------------
# Dispatcher for the "show" selector in the visualization notebook.
#
# Every option maps to a file to read and a function to display it with. The
# file names follow the same rule as _writer_keys() on the saver classes, so
# the options are resolved through the tables below.
# ---------------------------------------------------------------------------

# Feature group -> file name suffix
_RESULT_GROUP_SUFFIX = {
    "basic features": "",
    "differences of basic features between nodes": "_node_diff",
    "edge features": "_edge",
    "incident edge features per node": "_edge_per_node",
    "MST features": "_mst",
    "geometric features": "_geometric",
    "cluster number based on the distance between nodes": "_cluster",
    "configuration patterns based on the distance between nodes": "_config",
}

# Kind of result -> (file name template, how to display it)
_RESULT_KIND = {
    "Correlation": ("correlation{suffix}.xlsx", "correlation"),
    "Correlation in highlighted segments": ("correlation{suffix}_in_highlighted_segments.xlsx", "correlation"),
    "Hellinger distance in highlighted segments": ("hellinger_distance{suffix}.xlsx", "hellinger"),
    "Histogram in highlighted segments": ("histogram{suffix}.pkl", "histogram"),
}

# Groups whose file may legitimately be missing; report instead of raising.
_OPTIONAL_SUFFIXES = {"_geometric"}

# Histogram groups that reuse the caller's feature_name_list. The others take
# their feature names from the keys of the loaded dictionary.
_HISTOGRAM_USES_GIVEN_NAMES = {"", "_node_diff"}


def _is_multi_mode_option(show: str) -> bool:
    """Whether the option is marked "[Multi mode]", i.e. multi models only."""
    return show.replace("\u00a0", " ").strip().startswith("[Multi mode]")


def _parse_show_option(show: str) -> Tuple[str, str]:
    """Split "[Multi mode] Correlation (edge features)" into its kind and group.

    Some of the Colab dropdown options contain non-breaking spaces (U+00A0),
    so those are normalised to ordinary spaces first.
    """
    text = show.replace("\u00a0", " ").strip()
    text = re.sub(r"^\[Multi mode\]\s*", "", text)
    matched = re.match(r"^(.*?)\s*\((.*)\)$", text)
    if not matched:
        raise ValueError(f"Unrecognized option: {show!r}")
    return matched.group(1).strip(), matched.group(2).strip()


def show_selected_result(
    show,
    class_name_list,
    number_of_attention_branches=None,
    feature_name_list=None,
    dpi=100,
    model_type=None,
):
    """Load and display the result selected by the "show" dropdown.

    "[Multi mode]" options apply to multi models only; for a single model the
    reason is printed and nothing is displayed.
    """
    if model_type is not None and model_type != "multi" and _is_multi_mode_option(show):
        print(f'"{show}" is available only when model_type == "multi" (current: {model_type}).')
        return

    kind, group = _parse_show_option(show)
    if kind not in _RESULT_KIND:
        raise ValueError(f"Unrecognized option: {show!r}")
    if group not in _RESULT_GROUP_SUFFIX:
        raise ValueError(f"Unrecognized feature group: {group!r} (from {show!r})")

    template, how = _RESULT_KIND[kind]
    suffix = _RESULT_GROUP_SUFFIX[group]
    path = template.format(suffix=suffix)

    if suffix in _OPTIONAL_SUFFIXES and not os.path.isfile(path):
        print(f"{path} not found: (x, y) coordinates are required to build geometric features.")
        return

    def print_conditions():
        for i in range(len(class_name_list)):
            print("🟢 Condition", i + 1, ":", class_name_list[i])
        print()

    if how == "correlation":
        dictionary = pd.read_excel(path, sheet_name=None)
        show_correlation_table_and_gui(dictionary, class_name_list, number_of_attention_branches)

    elif how == "hellinger":
        print_conditions()
        dictionary = pd.read_excel(path, sheet_name=None)
        comparison_name_list = list(dictionary["Sheet1"].columns)
        comparison_name_list.remove("Feature")
        show_hellinger_distance_table_and_gui(dictionary, comparison_name_list)

    else:  # histogram
        with gzip.open(path, "rb") as file:
            dictionary = pickle.load(file)
        print_conditions()
        names = (
            feature_name_list
            if suffix in _HISTOGRAM_USES_GIVEN_NAMES
            else dictionary[list(dictionary.keys())[0]].keys()
        )
        show_distribution_of_handcrafted_features_and_gui(
            dictionary, class_name_list, names, dpi=dpi
        )


def draw_loss_curve(
    name_of_loss,
    save_filename,
    ylim=None,
):
    """Save one learning curve as a png.

    ylim=None autoscales the y axis; pass a (low, high) tuple to fix it.
    """
    plt.figure(figsize=(15, 3))
    left = list(range(0, len(name_of_loss)))
    height = name_of_loss
    plt.plot(left, height)
    plt.xticks(np.arange(0, len(name_of_loss), 1))
    plt.xticks(rotation=90)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig(f"{save_filename}.png")
    plt.close()


def draw_all_loss_curves(
    learning_utils_module,
):
    """Save all eight learning curves recorded during training."""
    draw_loss_curve(learning_utils_module.global_training_loss, "training_loss")
    draw_loss_curve(learning_utils_module.global_validation_loss, "validation_loss")
    draw_loss_curve(learning_utils_module.global_training_attention_distribution_penalty,
                    "training_attention_distribution_penalty")
    draw_loss_curve(learning_utils_module.global_validation_attention_distribution_penalty,
                    "validation_attention_distribution_penalty")
    draw_loss_curve(learning_utils_module.global_training_cross_entropy_losses,
                    "training_cross_entropy_losses", ylim=(0, 1))
    draw_loss_curve(learning_utils_module.global_training_similarities,
                    "training_similarities_of_attentions", ylim=(0, 1))
    draw_loss_curve(learning_utils_module.global_validation_cross_entropy_losses,
                    "validation_cross_entropy_losses", ylim=(0, 1))
    draw_loss_curve(learning_utils_module.global_validation_similarities,
                    "validation_similarities_of_attentions", ylim=(0, 1))


def find_angle_feature_indices(
    feature_name_list,
):
    """Return the indices of angular features, whose differences wrap around.

    Which features hold angles is defined by preprocessing.is_angular_feature(),
    so that the two modules agree.
    """
    return [
        i for i, name in enumerate(feature_name_list)
        if preprocessing.is_angular_feature(name)
    ]


def save_all_results(
    model_type,
    task_type,
    number_of_attention_branches,
    number_of_classes,
    has_pose,
    post_analysis_dictionary,
    prediction_tensor_list,
    attentions_tensor,
    test_label,
    save_folder,
):
    """Run the whole post-analysis and write its results to save_folder.

    Writes:

      - macro_averaged_accuracy.xlsx / classification_report.xlsx
      - prediction_class.xlsx / prediction_true_false.xlsx
      - prediction_attention_namelist.pkl
      - correlation*.xlsx (CorrelationSaver)
      - histogram*.pkl / hellinger_distance*.xlsx (DistributionSaver)
      - mean node attention (multi models only)
      - the data behind the interactive plots

    Returns the values needed for visualization, as a dict.
    """
    os.chdir(save_folder)

    # Save macro-averaged accuracy of each attention branch. Also get processed predictions and labels.
    prediction_list, label_list, prediction_argmax_list, label_argmax_list = save_macro_averaged_accuracy_table(
        prediction_tensor_list,
        test_label,
        task_type,
        number_of_attention_branches,
    )

    print("")
    print("")

    # Save classification report of each attention branch
    save_classification_report(
        prediction_argmax_list,
        label_argmax_list,
        task_type,
        number_of_attention_branches,
        post_analysis_dictionary["class_name_list"],
    )

    attentions = attentions_tensor.to("cpu").detach().numpy().copy()  # From GPU to CPU
    if model_type == "single":
        attentions = np.expand_dims(attentions, axis=2)

    predictions = np.stack(prediction_list, axis=1)
    predictions_argmax = np.stack(prediction_argmax_list, axis=1)
    labels_argmax = np.stack(label_argmax_list, axis=1)

    pred_masks = save_prediction_tables(
        num_attentions=number_of_attention_branches,
        num_classes=number_of_classes,
        post_analysis=post_analysis_dictionary,
        predictions=predictions_argmax,
        labels=labels_argmax,
    )

    data_to_save = {
        "attentions": attentions,
        "predictions": predictions,
        "prediction_argmax": predictions_argmax,
        "labels_argmax": labels_argmax,
        "pred_masks": pred_masks,
        "class_name_list": post_analysis_dictionary["class_name_list"],
        "feature_name_list": post_analysis_dictionary["feature_name_list"],
        "edge_data_name_list": post_analysis_dictionary['edge_data_name_list'],
    }
    with gzip.open("prediction_attention_namelist.pkl", "wb") as file:
        pickle.dump(data_to_save, file)

    angle_feature_indices = find_angle_feature_indices(post_analysis_dictionary['feature_name_list'])

    CorrelationSaver(
        model_type=model_type,
        post_analysis_dictionary=post_analysis_dictionary,
        attentions=attentions,
        pred_masks=pred_masks,
        angle_feature_indices=angle_feature_indices,
    ).save_all()

    DistributionSaver(
        model_type=model_type,
        post_analysis_dictionary=post_analysis_dictionary,
        attentions=attentions,
        pred_masks=pred_masks,
        angle_feature_indices=angle_feature_indices,
    ).save_all()

    if model_type == "multi":
        mean_node_attention(
            post_analysis_dictionary,
            attentions,
            pred_masks,
        )

    save_data_for_interactive_attention_plots(
        model_type,
        number_of_classes,
        has_pose,
        post_analysis_dictionary,
        attentions,
        predictions_argmax,
        labels_argmax,
    )

    return {
        "attentions": attentions,
        "predictions": predictions,
        "predictions_argmax": predictions_argmax,
        "labels_argmax": labels_argmax,
        "pred_masks": pred_masks,
        "angle_feature_indices": angle_feature_indices,
    }
