import os
import glob
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import copy


# Feature coherence penalty. A forward hook captures the input of every
# attention layer, and the penalty measures how much those features vary over
# the segments the attention actually weights. Disabled when
# LossConfig.lambda_for_feature_coherence is 0, which is the default.
_ATTENTION_CLASS_NAMES = ("MaskedAttentionBlock", "AttentionBlock")


class AttentionInputCapture:
    """Capture the input tensor of every attention layer during a forward pass.

        with AttentionInputCapture(model) as cap:
            predictions, attentions = model(x, ...)
        branch_attention_inputs = cap.get()   # one tensor per attention layer
    """

    def __init__(self, model, target_class_names=_ATTENTION_CLASS_NAMES):
        self.model = model
        self.target_names = set(target_class_names)
        self.captured = []
        self.handles = []

    def __enter__(self):
        self.captured.clear()
        for _name, module in self.model.named_modules():
            if module.__class__.__name__ in self.target_names:
                self.captured.append(None)
                slot = len(self.captured) - 1

                def make_hook(s):
                    def hook(_mod, inp, _out):
                        # inp is the tuple of positional args; the first one is the activation.
                        self.captured[s] = inp[0]
                    return hook

                self.handles.append(module.register_forward_hook(make_hook(slot)))
        return self

    def get(self):
        if any(c is None for c in self.captured):
            missing = [i for i, c in enumerate(self.captured) if c is None]
            raise RuntimeError(
                f"AttentionInputCapture: no input captured for attention slot(s) {missing}"
            )
        return list(self.captured)

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def feature_coherence_penalty(
    attentions,
    branch_features,
    threshold_scale=5.0,
    sigmoid_sharpness=5.0,
    eps=1e-8,
):
    """Penalise attention that spreads over features of differing values.

    The attention is normalised into a distribution, then a sigmoid gate
    softly removes the weights that fall below threshold_scale times the
    uniform weight. The penalty is the attention-weighted variance of the
    features around their attention-weighted mean, so a smaller value means
    the branch is attending to a segment that is homogeneous in feature space.

    Args:
        attentions: (B, num_branches, T) for single, (B, num_branches, N, T) for multi.
        branch_features: the attention-layer input of each branch,
            (B, D, T) for single and (B, D, N, T) for multi.
        threshold_scale: how many times the uniform weight the soft gate sits at.
        sigmoid_sharpness: sharpness of that gate.
    Returns:
        A scalar averaged over the batch and the branches.
    """
    values = []

    if attentions.dim() == 3:        # single
        for i, f in enumerate(branch_features):
            a = attentions[:, i, :]                                   # (B, T)
            a = a / (a.sum(dim=-1, keepdim=True) + eps)
            threshold = threshold_scale / a.shape[-1]
            a = a * torch.sigmoid(sigmoid_sharpness * (a / threshold - 1.0))
            a = a / (a.sum(dim=-1, keepdim=True) + eps)
            mu = (a.unsqueeze(1) * f).sum(dim=-1)                      # (B, D)
            diff = f - mu.unsqueeze(-1)                                # (B, D, T)
            squared = (diff ** 2).sum(dim=1)                           # (B, T)
            values.append((a * squared).sum(dim=-1).mean())

    elif attentions.dim() == 4:      # multi
        batch_size = attentions.shape[0]
        for i, f in enumerate(branch_features):
            a = attentions[:, i].reshape(batch_size, -1)               # (B, N*T)
            a = a / (a.sum(dim=-1, keepdim=True) + eps)
            threshold = threshold_scale / a.shape[-1]
            a = a * torch.sigmoid(sigmoid_sharpness * (a / threshold - 1.0))
            a = a / (a.sum(dim=-1, keepdim=True) + eps)
            channels = f.shape[1]
            flat = f.reshape(batch_size, channels, -1)                 # (B, D, N*T)
            mu = (a.unsqueeze(1) * flat).sum(dim=-1)                   # (B, D)
            diff = flat - mu.unsqueeze(-1)
            squared = (diff ** 2).sum(dim=1)                           # (B, N*T)
            values.append((a * squared).sum(dim=-1).mean())

    else:
        raise ValueError(f"unexpected attention shape: {attentions.shape}")

    return sum(values) / len(values)


@dataclass
class LossConfig:
    """Hyperparameters of the loss.

    Shared by compute_batch_loss, compute_validation_loss and
    train_and_validate.
    """

    task_type: str
    number_of_attention_branches: int
    C_CE: float
    C_S: float
    threshold_of_loss_penalty: Any  # a number, or the string "Default"
    lambda_for_total_variation_regularization: Any  # a number, or the string "Default"
    penalty_weight_for_attention_distribution: float
    threshold_of_attention_distribution: float
    # Feature coherence penalty; 0 disables it.
    lambda_for_feature_coherence: float = 0.0
    feature_coherence_threshold_scale: float = 5.0   # 10.0 for the stricter setting
    feature_coherence_sigmoid_sharpness: float = 5.0


@dataclass
class ModelSpec:
    """Everything needed to build a model.

    The number_of_* fields come from load_data(); the rest are user settings.
    number_of_nodes, number_of_dimensions_of_edge_weight, edge_index,
    stgcn_out_channels, node_shuffle and max_hop are unused for single models.
    """

    model_type: str
    number_of_classes: int
    number_of_channels: int
    number_of_timestamps: int
    number_of_attention_branches: int
    number_of_channels_of_pose: int = 0
    number_of_nodes: int = 0
    number_of_dimensions_of_edge_weight: int = 0
    edge_index: Any = None
    conv1d_out_channels: int = 64
    lstm_hidden_size: int = 64
    stgcn_out_channels: int = 64
    node_shuffle: bool = True
    max_hop: int = 1


@dataclass
class ValidationData:
    """The whole validation set as tensors, alongside its DataLoader.

    Used to compute the macro averaged accuracy over the full validation set
    once per epoch. edge_weight is unused for single models.
    """

    x: Any
    label: Any
    edge_weight: Any
    x_pose: Any

global_training_loss = []
global_training_cross_entropy_losses = [] # cross entropy losses from all attention branches
global_training_similarities = []  # similarities of attentions
global_training_attention_distribution_penalty = []
global_validation_loss = []
global_validation_cross_entropy_losses = []  # cross entropy losses from all attention branches
global_validation_similarities = []  # similarities of attentions
global_validation_attention_distribution_penalty = []
global_training_feature_coherence = []      # stays 0 while the penalty is disabled
global_validation_feature_coherence = []


def load_data(
    set_names, 
    train_val_test_dictionary, 
    model_type, 
    batch_size, 
    DEVICE,
):
    print(model_type)

    if set_names == "train_and_val":
        set_names = ["train", "val"]
    if set_names == "test":
        set_names = ["test"]

    def process_data(data, dtype, squeeze=False):
        if np.all(data == None):
            data = torch.full(data.shape, float("nan"))  # convert None to nan because torch can not treat None
        else:
            data = torch.from_numpy(data).type(dtype).to(DEVICE)
        if squeeze:
            data = data.squeeze(dim=2)
        return data

    # Process data
    dataloaders = []
    has_pose = False

    for set_name in set_names:

        # Load data
        x = train_val_test_dictionary["features"][set_name]
        label = train_val_test_dictionary["labels"][set_name]
        edge_index = train_val_test_dictionary["edge_index"][set_name]
        edge_weight = train_val_test_dictionary["edge_attributes"][set_name]
        x_pose = train_val_test_dictionary["pose_feature"][set_name]

        # The data has pose?
        if np.all(x_pose == None):
            has_pose = False
        else:
            has_pose = True

        # Process data
        x = process_data(x, torch.FloatTensor, squeeze=(model_type == "single"))
        label = process_data(label, torch.FloatTensor)
        edge_index = process_data(edge_index, torch.LongTensor)
        edge_weight = process_data(edge_weight, torch.FloatTensor)
        x_pose = process_data(x_pose, torch.FloatTensor, squeeze=((model_type == "single") and has_pose))

        if len(edge_weight.shape) == 5:  # for multi
            x = x.permute(0, 1, 3, 2)
                    
        if len(x_pose.shape) == 3:  # for single
            x_pose = x_pose.permute(0, 2, 1)

        if len(x_pose.shape) == 4:  # for multi
            x_pose = x_pose.permute(0, 2, 3, 1)

        print(set_name, "set")
        print("x:", x.shape)
        print("label:", label.shape)
        print("edge_index:", edge_index[0].T.shape)
        print("edge_weight:", edge_weight.shape)
        print("x_pose:", x_pose.shape)
        print("has_pose:", has_pose)

        if set_name == "train":
            if model_type == "single":
                dataset = torch.utils.data.TensorDataset(x, label, x_pose)
            if model_type == "multi":
                dataset = torch.utils.data.TensorDataset(x, label, edge_weight, x_pose)                
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        if set_name == "val":
            if model_type == "single":
                dataset = torch.utils.data.TensorDataset(x, label, x_pose)
            if model_type == "multi":
                dataset = torch.utils.data.TensorDataset(x, label, edge_weight, x_pose)                
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

        if set_name == "test":
            loader = []
        
        dataloaders.append(loader)  # final length 2. train loader, val loader

    # Get the values to load the model
    if model_type == "single":
        _, number_of_channels, number_of_timestamps = x.shape
        _, number_of_classes = label.shape
        number_of_nodes = 0
        number_of_dimensions_of_edge_weight = 0
    if model_type == "multi":
        _, number_of_channels, number_of_timestamps, number_of_nodes = x.shape
        _, number_of_classes = label.shape
        *_, number_of_dimensions_of_edge_weight = edge_weight.shape

    if has_pose:
        *_, number_of_channels_of_pose = x_pose.shape
    else:
        number_of_channels_of_pose = 0

    if has_pose:
        number_of_channels += 1

    print("number_of_classes =", number_of_classes)
    print("number_of_channels =", number_of_channels)
    print("number_of_timestamps =", number_of_timestamps)
    print("number_of_nodes =", number_of_nodes)
    print("number_of_dimensions_of_edge_weight =", number_of_dimensions_of_edge_weight)
    print("number_of_channels_of_pose =", number_of_channels_of_pose)

    return dataloaders, number_of_classes, number_of_channels, number_of_timestamps, number_of_nodes, number_of_dimensions_of_edge_weight, number_of_channels_of_pose, x, label, edge_index[0].T, edge_weight, x_pose, has_pose


def read_optuna_summary(
    text,
):
    branches = {}
    lines = text.strip().split("\n")
    for line in lines:
        if not line.startswith("branch"):
            continue
        key_value = line.strip().split(": ")
        if len(key_value) != 2:
            continue
        key, value = key_value

        parts = key.split("_")
        branch_key = "_".join(parts[:2])  # eg.,（branch_0）
        param_name = "_".join(parts[2:])  # eg.,（MaskedConv1dBlock）

        branch_index = int(branch_key.replace("branch_", ""))
        value = int(value)
        if branch_index not in branches:
            branches[branch_index] = {}
        branches[branch_index][param_name] = value

    branch_list = []
    for i in sorted(branches.keys()):
        branch_list.append(branches[i])

    return branch_list


def read_key_value_file(
    path,
    skip_malformed_lines=False,
):
    """Read a "key: value" text file into a dict.

    optuna_summary.txt also contains lines that are not "key: value", so pass
    skip_malformed_lines=True when reading it.
    """
    loaded_values = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(": ")
            if len(parts) != 2:
                if skip_malformed_lines:
                    continue
                raise ValueError(f'{path}: cannot parse line as "key: value": {line.strip()!r}')
            key, parameter = parts
            loaded_values[key] = parameter
    return loaded_values


def read_learning_settings(
    path="learning_settings.txt",
):
    """Read learning_settings.txt and convert each value to its type.

    The architecture search writes the file; training and testing read it back.
    """
    raw = read_key_value_file(path)
    return {
        "model_type": raw["model_type"],
        "utils_folder": raw["utils_folder"],
        # Name of the utils_YYYYMMDDHHMMSS folder inside the project. Absent
        # from files written before projects carried a snapshot.
        "utils_snapshot": raw.get("utils_snapshot"),
        "dataset_folder": raw["dataset_folder"],
        "task_type": raw["task_type"],
        "number_of_attention_branches": int(raw["number_of_attention_branches"]),
        "seed": int(raw["seed"]),
        "batch_size": int(raw["batch_size"]),
        "learning_rate": float(raw["learning_rate"]),
        "C_CE": float(raw["C_CE"]),
        "C_S": float(raw["C_S"]),
        # Older files may not have this key.
        "threshold_of_attention_distribution": float(raw.get("threshold_of_attention_distribution", 0.8)),
        # May be the string "Default", so it is left as it is.
        "threshold_of_loss_penalty": raw["threshold_of_loss_penalty"],
        "lambda_for_total_variation_regularization": raw["lambda_for_total_variation_regularization"],
        "conv1d_out_channels": int(raw["conv1d_out_channels"]),
        "lstm_hidden_size": int(raw["lstm_hidden_size"]),
        "stgcn_out_channels": int(raw["stgcn_out_channels"]),
        "node_shuffle": raw["node_shuffle"] == "True",
        # Feature coherence penalty. Defaults keep older files readable;
        # a lambda of 0 leaves the penalty switched off.
        "lambda_for_feature_coherence": float(raw.get("lambda_for_feature_coherence", 0.0)),
        "feature_coherence_threshold_scale": float(raw.get("feature_coherence_threshold_scale", 5.0)),
        "feature_coherence_sigmoid_sharpness": float(raw.get("feature_coherence_sigmoid_sharpness", 5.0)),
    }


def read_optuna_summary_settings(
    model_type,
    path="optuna_summary.txt",
):
    """Read the non-branch hyperparameters from optuna_summary.txt.

    read_optuna_summary() reads the per-branch architecture from the same file.
    """
    raw = read_key_value_file(path, skip_malformed_lines=True)
    settings = {
        "dropout_probability": float(raw["dropout_probability"]),
        "weight_decay": float(raw["weight_decay"]),
        "temporal_kernel_size_before_branching": int(raw["temporal_kernel_size_before_branching"]),
        "macro_averaged_accuracies": [eval(raw["macro_averaged_accuracies"])],
        "similarities": [eval(raw["similarities"])],
    }
    if model_type == "single":
        settings["before_branching_blocks"] = int(raw["MaskedConv1dBlock_before_branching"])
    elif model_type == "multi":
        settings["before_branching_blocks"] = int(raw["STGCNBlock_before_branching"])
    else:
        raise ValueError(f'model_type must be "single" or "multi", got {model_type!r}')
    return settings


def write_preset_hyperparameters(
    preset_hyperparameters,
    optuna_folder,
):
    """Write a ready-made architecture to optuna_summary.txt.

    Used when the architecture search is skipped; the values come from the
    caller so that they are easy to edit.
    """
    learning_file = Path(optuna_folder) / "optuna_summary.txt"
    if learning_file.exists():
        os.remove(learning_file)

    with open(learning_file, "a") as file:
        for key, parameter in preset_hyperparameters.items():
            file.write(f"{key}: {parameter}\n")

    return preset_hyperparameters


def write_optuna_summary(
    study,
    path="optuna_summary.txt",
):
    """Write optuna_summary.txt from a study, for notebooks 03 and 04 to read."""
    import optuna

    if os.path.exists(path):
        os.remove(path)

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    with open(path, "a") as file:
        file.write("Study statistics: \n")
        file.write("    Number of finished trials: {}\n".format(len(study.trials)))
        file.write("    Number of pruned trials: {}\n".format(len(pruned_trials)))
        file.write("    Number of complete trials: {}\n".format(len(complete_trials)))
        file.write("Best trial: {}\n".format(study.best_trial.number))
        for key, value in study.best_trial.params.items():
            file.write("{}: {}\n".format(key, value))
        file.write("loss: {}\n".format(study.best_trial.value))
        file.write("cross_entropy_losses: {}\n".format(study.best_trial.user_attrs["cross_entropy_losses"]))
        file.write("similarities: {}\n".format(study.best_trial.user_attrs["similarities"]))
        file.write("attention_distribution_penalty: {}\n".format(study.best_trial.user_attrs["attention_distribution_penalty"]))
        file.write("macro_averaged_accuracies: {}\n".format(study.best_trial.user_attrs["macro_averaged_accuracies"]))


def run_optuna_study(
    objective,
    number_of_optuna_trials,
    seed,
    study_path="study.pkl",
):
    """Run the architecture search, resuming from study.pkl when it exists.

    study.pkl and optuna_summary.txt are written to the current directory.
    optuna and joblib are imported inside this function so that notebooks 03
    and 04 can use learning_utils without having optuna installed.
    """
    import joblib
    import optuna

    def save_study(study, trial):
        joblib.dump(study, study_path)
        print("Study saved at trial {}".format(trial.number))
        write_optuna_summary(study)

    if os.path.exists(study_path):
        study = joblib.load(study_path)
        print("Resuming study from saved state.")
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler, direction="minimize")
        print("Starting a new study.")

    study.optimize(
        lambda trial: objective(trial),
        n_trials=number_of_optuna_trials,
        callbacks=[save_study],
    )

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    print("  Trial number: ", study.best_trial.number)
    print("  Value: ", study.best_trial.value)

    return study


def run_inference(
    model,
    model_type,
    test_x,
    test_edge_weight,
    test_x_pose,
    has_pose,
):
    """Run the model over the test set and return its predictions and attentions.

    edge_weight is unused for single models.
    """
    model.eval()
    with torch.no_grad():
        if model_type == "single":
            prediction_tensor_list, attentions_tensor = model(
                test_x,
                test_x_pose,
                has_pose,
            )
        elif model_type == "multi":
            prediction_tensor_list, attentions_tensor = model(
                test_x,
                test_edge_weight,
                test_x_pose,
                has_pose,
            )
        else:
            raise ValueError(f'model_type must be "single" or "multi", got {model_type!r}')

    # prediction_tensor_list: number_of_attention_branches tensors of (number_of_files, number_of_classes)
    # attentions_tensor: (number_of_files, number_of_attention_branches, number_of_timestamps)
    return prediction_tensor_list, attentions_tensor


def setup_determinism(
    seed,
    cuda="cuda:0",
):
    """Seed every random generator, switch on deterministic mode, return the device.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    use_cuda = torch.cuda.is_available()
    assert use_cuda, "Change the runtime to GPU"
    device = torch.device(cuda)
    print("CUDA:", use_cuda, device)

    return device


def ensure_odd(
    number_of_timestamps,
    value,
):
    """Round a fraction of the sequence length to the nearest odd kernel size."""
    result = int(number_of_timestamps * value)
    if result % 2 == 0:
        result += 3
    return result


def build_configs(
    spec,
    dropout_probability,
    before_branching_blocks,
    temporal_kernel_size_before_branching,
    branch_list,
):
    """Build before_branch_config and branch_config.

    branch_list holds one dict per branch, as returned by read_optuna_summary()
    and by suggest_architecture(). Its keys are MaskedConv1dBlock,
    MaskedLSTMBlock, attention_position and temporal_kernel_size for single
    models, and STGCNBlock, attention_position and temporal_kernel_size for
    multi models.
    """
    if spec.model_type == "single":
        before_branch_config = generate_before_branch_config_for_singlemodel(
            spec.number_of_channels,
            spec.conv1d_out_channels,
            dropout_probability,
            before_branching_blocks,
            temporal_kernel_size_before_branching,
        )
    elif spec.model_type == "multi":
        before_branch_config = generate_before_branch_config_for_multimodel(
            spec.number_of_channels,
            spec.stgcn_out_channels,
            dropout_probability,
            before_branching_blocks,
            temporal_kernel_size_before_branching,
        )
    else:
        raise ValueError(f'model_type must be "single" or "multi", got {spec.model_type!r}')

    # Fall back to a fixed in_channels when nothing runs before the branches.
    if before_branch_config and len(before_branch_config) > 0:
        first_in_channels = before_branch_config[-1]["out_channels"]
    else:
        first_in_channels = copy.deepcopy(spec.number_of_channels)

    branch_config = []
    for branch_info in branch_list:
        if spec.model_type == "single":
            branch_cnfg = generate_branch_cnfg_for_singlemodel(
                spec.number_of_timestamps,
                first_in_channels,
                spec.conv1d_out_channels,
                spec.lstm_hidden_size,
                dropout_probability,
                branch_info["MaskedConv1dBlock"],
                branch_info["MaskedLSTMBlock"],
                branch_info["attention_position"],
                branch_info["temporal_kernel_size"],
            )
        else:
            branch_cnfg = generate_branch_cnfg_for_multimodel(
                spec.number_of_nodes,
                spec.number_of_timestamps,
                first_in_channels,
                spec.stgcn_out_channels,
                dropout_probability,
                branch_info["STGCNBlock"],
                branch_info["attention_position"],
                branch_info["temporal_kernel_size"],
            )
        branch_config.append(branch_cnfg)

    return before_branch_config, branch_config


def build_model(
    model_module,
    spec,
    before_branch_config,
    branch_config,
    device,
):
    """Build either model and move it to the device, with one call signature."""
    if spec.model_type == "single":
        model = model_module.Model(
            before_branch_config=before_branch_config,
            branch_config=branch_config,
            output_size=spec.number_of_classes,
            in_channels_pose=spec.number_of_channels_of_pose,
        )
    elif spec.model_type == "multi":
        model = model_module.Model(
            before_branch_config=before_branch_config,
            branch_config=branch_config,
            output_size=spec.number_of_classes,
            in_channels=spec.number_of_channels,
            in_channels_pose=spec.number_of_channels_of_pose,
            in_channels_edge_weight=spec.number_of_dimensions_of_edge_weight,
            edge_index=spec.edge_index,
            num_node=spec.number_of_nodes,
            node_shuffle=spec.node_shuffle,
            max_hop=spec.max_hop,
        )
    else:
        raise ValueError(f'model_type must be "single" or "multi", got {spec.model_type!r}')

    return model.to(device)


def suggest_architecture(
    trial,
    spec,
    userset_length_to_calculate_kernel_size=None,
):
    """Suggest one architecture from an Optuna trial, for either model type."""
    length = userset_length_to_calculate_kernel_size or spec.number_of_timestamps

    def suggest_kernel_size(name):
        return trial.suggest_int(
            name,
            ensure_odd(length, 0.01),
            ensure_odd(length, 0.10),
            step=2,
        )

    dropout_probability = trial.suggest_float("dropout_probability", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    ##### before branching #####
    if spec.model_type == "single":
        before_branching_blocks = trial.suggest_int("MaskedConv1dBlock_before_branching", 0, 3)
    elif spec.model_type == "multi":
        before_branching_blocks = trial.suggest_int("STGCNBlock_before_branching", 0, 3)
    else:
        raise ValueError(f'model_type must be "single" or "multi", got {spec.model_type!r}')

    temporal_kernel_size_before_branching = suggest_kernel_size("temporal_kernel_size_before_branching")

    ##### branching #####
    branch_list = []
    for branch_idx in range(spec.number_of_attention_branches):
        if spec.model_type == "single":
            # 0-5 MaskedConv1dBlocks and 0-5 MaskedLSTMBlocks per branch. The
            # AttentionBlock can sit before, between or after the conv blocks.
            number_of_MaskedConv1dBlock = trial.suggest_int(f"branch_{branch_idx}_MaskedConv1dBlock", 0, 5)
            number_of_MaskedLSTMBlock = trial.suggest_int(f"branch_{branch_idx}_MaskedLSTMBlock", 0, 5)
            attention_position = trial.suggest_int(
                f"branch_{branch_idx}_attention_position", 0, number_of_MaskedConv1dBlock
            )
            branch_info = {
                "MaskedConv1dBlock": number_of_MaskedConv1dBlock,
                "MaskedLSTMBlock": number_of_MaskedLSTMBlock,
                "attention_position": attention_position,
            }
        else:
            # 0-7 STGCNBlocks per branch; the AttentionBlock can sit anywhere among them.
            number_of_STGCNBlock = trial.suggest_int(f"branch_{branch_idx}_STGCNBlock", 0, 7)
            attention_position = trial.suggest_int(
                f"branch_{branch_idx}_attention_position", 0, number_of_STGCNBlock
            )
            branch_info = {
                "STGCNBlock": number_of_STGCNBlock,
                "attention_position": attention_position,
            }

        branch_info["temporal_kernel_size"] = suggest_kernel_size(
            f"branch_{branch_idx}_temporal_kernel_size"
        )
        branch_list.append(branch_info)

    return {
        "dropout_probability": dropout_probability,
        "weight_decay": weight_decay,
        "before_branching_blocks": before_branching_blocks,
        "temporal_kernel_size_before_branching": temporal_kernel_size_before_branching,
        "branch_list": branch_list,
    }


def calculate_hyperparameters(
    threshold_of_loss_penalty,
    lambda_for_total_variation_regularization,
    C_CE,
    C_S,
    number_of_classes,
    number_of_attention_branches,
):

    """
    threshold_of_loss_penalty:
        In N-class classification,
        when the class distribution is uniform and the model's predictions are completely random (probability 1/N for each class),
        the cross-entropy loss is equal to log(N) regardless of the true label.

    lambda_for_total_variation_regularization:
        eg., in 2-class classification,
        {(loss[0.69*0.6]) + C_CE[3]) * (similarity[0.5] + C_S[1])} * branches[10] = 51.21  # 0.6: accurate classification
        51.21 * 0.4 = 20.484  # 0.4: not too strong and not too weak.

    penalty_weight_for_attention_distribution:
        The calculation is same as above.
    """

    if threshold_of_loss_penalty == "Default":
        threshold_of_loss_penalty = np.log(number_of_classes) * 0.8  # 0.8: slightly acculate classification
    print("threshold_of_loss_penalty =", threshold_of_loss_penalty)


    if lambda_for_total_variation_regularization == "Default":
        lambda_for_total_variation_regularization = ((np.log(number_of_classes) * 0.6) + C_CE) * (0.5 + C_S) * number_of_attention_branches * 0.4
    print("lambda_for_total_variation_regularization =", lambda_for_total_variation_regularization)


    penalty_weight_for_attention_distribution = ((np.log(number_of_classes) * 0.6) + C_CE) * (0.5 + C_S) * number_of_attention_branches * 0.4
    print("penalty_weight_for_attention_distribution =", penalty_weight_for_attention_distribution)


    return threshold_of_loss_penalty, lambda_for_total_variation_regularization, penalty_weight_for_attention_distribution


def generate_before_branch_config_for_singlemodel(
    number_of_channels,
    conv1d_out_channels,
    dropout_probability,
    MaskedConv1dBlock_before_branching,
    temporal_kernel_size_before_branching,

):

    before_branch_config = []

    # If there is at least one MaskedConv1dBlock, set in_channels and out_channels
    in_channels = copy.deepcopy(number_of_channels)

    for i in range(MaskedConv1dBlock_before_branching):

        # Add the MaskedConv1dBlock configuration
        block_config = {
            "layertype": "MaskedConv1dBlock",
            "in_channels": in_channels,
            "out_channels": conv1d_out_channels,
            "temporal_kernel_size": temporal_kernel_size_before_branching,
            "dropout_probability": dropout_probability,
        }

        # Append to before_branch_config
        before_branch_config.append(block_config)

        # Update in_channels for the next block
        in_channels = conv1d_out_channels

    return before_branch_config


def generate_branch_cnfg_for_singlemodel(
    number_of_timestamps,
    in_channels,
    conv1d_out_channels,
    lstm_hidden_size,
    dropout_probability,
    number_of_Conv1dBlock,
    number_of_LSTMBlock,
    attention_position,
    temporal_kernel_size_in_branch,
):

    branch_cnfg = []

    for i in range(number_of_Conv1dBlock):
        if i == attention_position:
            # Insert MaskedAttentionBlock at the specified position
            attention_block_config = {
                "layertype": "MaskedAttentionBlock",
                "in_channels": in_channels,
            }
            branch_cnfg.append(attention_block_config)

        # Add MaskedConv1dBlock configuration
        block_config = {
            "layertype": "MaskedConv1dBlock",
            "in_channels": in_channels,
            "out_channels": conv1d_out_channels,
            "temporal_kernel_size": temporal_kernel_size_in_branch,
            "dropout_probability": dropout_probability,
        }

        # Append to branch_cnfg
        branch_cnfg.append(block_config)

        # Update in_channels for the next block
        in_channels = conv1d_out_channels

    # After all MaskedConv1dBlocks, insert the MaskedAttentionBlock if it hasn't been added yet
    if attention_position == number_of_Conv1dBlock:
        attention_block = {
            "layertype": "MaskedAttentionBlock",
            "in_channels": in_channels,
        }
        branch_cnfg.append(attention_block)

    # Add MaskedLSTMBlock(s) before OutputBlock
    for i in range(number_of_LSTMBlock):
        # Add MaskedLSTMBlock configuration
        lstm_block_config = {
            "layertype": "MaskedLSTMBlock",
            "input_size": in_channels,
            "hidden_size": lstm_hidden_size,
            "dropout_probability": dropout_probability,
        }

        # Append to branch_cnfg
        branch_cnfg.append(lstm_block_config)

        # Update in_channels for the next block (LSTM is bidirectional, so multiply by 2)
        in_channels = lstm_hidden_size * 2

    # Add OutputBlock
    in_features = in_channels * number_of_timestamps
    output_block_config = {
        "layertype": "OutputBlock",
        "in_features": in_features,
    }
    branch_cnfg.append(output_block_config)

    return branch_cnfg


def generate_before_branch_config_for_multimodel(
    number_of_channels,
    stgcn_out_channels,
    dropout_probability,
    STGCNBlock_before_branching,
    temporal_kernel_size_before_branching,
):

    before_branch_config = []

    # If there is at least one STGCNBlock, set in_channels and out_channels
    in_channels = copy.deepcopy(number_of_channels)

    for i in range(STGCNBlock_before_branching):

        # Add the STGCNBlock configuration
        block_config = {
            "layertype": "STGCNBlock",
            "in_channels": in_channels,
            "out_channels": stgcn_out_channels,
            "temporal_kernel_size": temporal_kernel_size_before_branching,
            "dropout_probability": dropout_probability,
        }

        # Append to before_branch_config
        before_branch_config.append(block_config)

        # Update in_channels for the next block
        in_channels = stgcn_out_channels

    return before_branch_config


def generate_branch_cnfg_for_multimodel(
    number_of_nodes,
    number_of_timestamps,
    in_channels,
    stgcn_out_channels,
    dropout_probability,
    number_of_STGCNBlock,
    attention_position,
    temporal_kernel_size_in_branch,
):

    branch_cnfg = []

    for i in range(number_of_STGCNBlock):
        if i == attention_position:
            # Insert AttentionBlock at the specified position
            attention_block = {
                "layertype": "AttentionBlock",
                "in_channels": in_channels,
            }
            branch_cnfg.append(attention_block)

        # Add STGCNBlock configuration
        block_config = {
            "layertype": "STGCNBlock",
            "in_channels": in_channels,
            "out_channels": stgcn_out_channels,
            "temporal_kernel_size": temporal_kernel_size_in_branch,
            "dropout_probability": dropout_probability
        }

        # Append to branch_cnfg
        branch_cnfg.append(block_config)

        # Update in_channels for the next block
        in_channels = stgcn_out_channels

    # After all STGCNBlocks, insert the AttentionBlock if it hasn't been added yet
    if attention_position == number_of_STGCNBlock:
        attention_block_config = {
            "layertype": "AttentionBlock",
            "in_channels": in_channels,
        }
        branch_cnfg.append(attention_block_config)

    # Add OutputBlock
    in_features = in_channels * number_of_nodes * number_of_timestamps
    output_block_config = {
        "layertype": "OutputBlock",
        "in_features": in_features,
    }
    branch_cnfg.append(output_block_config)

    return branch_cnfg


def similarity_of_attentions(
    attentions,
):

    batch_size, number_of_attention_branches, *_ = attentions.shape
    averaged_similarities = torch.zeros(number_of_attention_branches)

    for i in range(number_of_attention_branches):
        similarities = []
        for j in range(number_of_attention_branches):
            if i != j:
                # Get attention matrices
                if len(attentions.shape) == 3:  # for single
                    attention_a = attentions[:, i, :]  # (B, T)
                    attention_b = attentions[:, j, :]  # (B, T)
                if len(attentions.shape) == 4:  # for multi
                    attention_a = attentions[:, i, :, :].reshape(batch_size, -1)  # (B, N*T)
                    attention_b = attentions[:, j, :, :].reshape(batch_size, -1)  # (B, N*T)

                # Calculate dot product
                dot_product = (attention_a * attention_b).sum(dim=1)

                # Calculate norm
                norm_a = attention_a.norm(dim=1)
                norm_b = attention_b.norm(dim=1)

                # Compute cosine similarity by dividing the dot product by the norms.
                # Add 1.0 and divide by 2 to shift the similarity range from -1 - 1 to 0 - 1
                similarity = (dot_product / (norm_a * norm_b) + torch.tensor(1.0, requires_grad=True)) / torch.tensor(2.0, requires_grad=True)
                similarities.append(similarity.mean())
        averaged_similarities[i] = torch.mean(torch.stack(similarities))

    return averaged_similarities  # Value range 0-1. [simirality, simirality...]. A list of length number_of_attention_branches.


def loss_penalty_with_threshold(
    loss,
    threshold_of_loss_penalty,
    small_loss_weight=1.0,  # float
    large_loss_weight=10.0,   # float
):
    """
    Calculation of loss penalty.
        eg. loss A from attention branch a.
            A_weight = small_loss_weight if A <= threshold else large_loss_weight
            weighted_loss = A_weight * A
    """

    small_loss_weight_tensor = torch.tensor(small_loss_weight, requires_grad=True)
    large_loss_weight_tensor = torch.tensor(large_loss_weight, requires_grad=True)

    weight = small_loss_weight_tensor if loss <= torch.tensor(threshold_of_loss_penalty) else large_loss_weight_tensor
    penaltied_loss = loss * weight

    return penaltied_loss


def total_variation_regularization(
    attentions,
    lambda_for_total_variation_regularization,
):
    
    # (the elements from the second index up to the last index) - (the elements from the first index up to the second-to-last index)
    regularization_term = torch.tensor(lambda_for_total_variation_regularization, requires_grad=True) * torch.sum(torch.abs(attentions[..., 1:] - attentions[..., :-1]), dim=-1)

    regularization_term = torch.mean(regularization_term)

    return regularization_term


def distribution_of_attention(
    attentions,
    penalty_weight_for_attention_distribution,
    threshold_of_attention_distribution,
):
    """
    Returns attention_distribution_penalty based on the distribution of values in attentions.
    If the attention values are excessively concentrated at the beginning and end segments, 
    the checker returns 1. Otherwise, it returns 0.
    """

    if len(attentions.shape) == 3:  # for single
        _, number_of_attention_branches, number_of_timestamps = attentions.shape
    if len(attentions.shape) == 4:  # for multi
        _, number_of_attention_branches, _, number_of_timestamps = attentions.shape

    mean_attention = torch.mean(attentions, dim=0)  # (B, number_of_attention_branches, (N), T) -average> (number_of_attention_branches, (N), T)

    attention_distribution_checker = torch.tensor(0.0, requires_grad=True)
    for i in range(number_of_attention_branches):
        if len(attentions.shape) == 3:  # for single
            if torch.sum(mean_attention[i, :int(number_of_timestamps*0.05)]) + torch.sum(mean_attention[i, int(number_of_timestamps*0.95):]) >= threshold_of_attention_distribution:
                checker = torch.tensor(1.0, requires_grad=True)
            else:
                checker = torch.tensor(0.0, requires_grad=True)
        if len(attentions.shape) == 4:  # for multi
            if torch.sum(mean_attention[i, :, :int(number_of_timestamps*0.05)]) + torch.sum(mean_attention[i, :, int(number_of_timestamps*0.95):]) >= threshold_of_attention_distribution:
                checker = torch.tensor(1.0, requires_grad=True)
            else:
                checker = torch.tensor(0.0, requires_grad=True)
        attention_distribution_checker = attention_distribution_checker + checker  # 0 <= attention_distribution_checker <= number_of_attention_branches

    attention_distribution_penalty = attention_distribution_checker * torch.tensor(float(number_of_attention_branches), requires_grad=True) * torch.tensor(penalty_weight_for_attention_distribution, requires_grad=True)

    return attention_distribution_penalty


def calculate_macro_averaged_accuracy(
    prediction,
    label, 
):
    # Get the unique class labels from both prediction and actual labels
    unique_labels = torch.unique(torch.cat((label, prediction)))
    
    # List to store accuracy for each class
    accuracies = []
    
    # Calculate accuracy for each class
    for u in unique_labels:
        # Indices for actual labels of the current class
        true_indices = (label == u)
        
        # Indices for predicted labels of the current class
        pred_indices = (prediction == u)
        
        # Count of correct predictions
        correct_predictions = torch.sum(true_indices & pred_indices).float()
        
        # Total number of data points for the current class
        total_true = torch.sum(true_indices).float()
        
        # Calculate accuracy for the current class (only if denominator is not zero)
        if total_true > 0:
            accuracy = correct_predictions / total_true
            accuracies.append(accuracy)
    
    # Calculate macro averaged accuracy
    macro_averaged_accuracy = torch.mean(torch.stack(accuracies))
    
    return macro_averaged_accuracy


def process_over_branches(
    predictions,
    label,
    task_type,
    number_of_attention_branches,
    func,
    use_argmax_for_func,
):
    
    if task_type == "classification":

        prediction_list = predictions
        label_list = [label for _ in range(number_of_attention_branches)]

    prediction_argmax_list = []
    label_argmax_list = []
    values = []

    for i in range(number_of_attention_branches):

        prediction_argmax = torch.argmax(prediction_list[i], dim=1)
        label_argmax = torch.argmax(label_list[i], dim=1)

        if use_argmax_for_func:
            value = func(prediction_argmax, label_argmax)
        else:
            value = func(prediction_list[i], label_list[i])

        prediction_argmax_list.append(prediction_argmax)
        label_argmax_list.append(label_argmax)
        values.append(value)

        # for debug
        # if i==0:
        #     print(label_argmax[0:10])
        #     print(prediction_argmax[0:10])

    return values, prediction_list, label_list, prediction_argmax_list, label_argmax_list


def forward_batch(
    model,
    model_type,
    batch_data,
    has_pose,
    capture_attention_inputs=False,
):
    """Run one batch through the model, optionally capturing the attention inputs.

    Returns (label, predictions, attentions, branch_attention_inputs); the last
    one is None unless capture_attention_inputs is set.
    """
    if model_type == "single":
        x, label, x_pose = batch_data
        args = (x, x_pose, has_pose)
    elif model_type == "multi":
        x, label, edge_weight, x_pose = batch_data
        args = (x, edge_weight, x_pose, has_pose)
    else:
        raise ValueError(f'model_type must be "single" or "multi", got {model_type!r}')

    if capture_attention_inputs:
        with AttentionInputCapture(model) as cap:
            batch_predictions, batch_attentions = model(*args)
        return label, batch_predictions, batch_attentions, cap.get()

    batch_predictions, batch_attentions = model(*args)
    return label, batch_predictions, batch_attentions, None


def compute_batch_loss(
    batch_predictions,
    batch_attentions,
    label,
    criterion,
    loss_config,
    branch_attention_inputs=None,
):

    task_type = loss_config.task_type
    number_of_attention_branches = loss_config.number_of_attention_branches
    C_CE = loss_config.C_CE
    C_S = loss_config.C_S
    threshold_of_loss_penalty = loss_config.threshold_of_loss_penalty
    lambda_for_total_variation_regularization = loss_config.lambda_for_total_variation_regularization
    penalty_weight_for_attention_distribution = loss_config.penalty_weight_for_attention_distribution
    threshold_of_attention_distribution = loss_config.threshold_of_attention_distribution

    if task_type == "classification":

        # Compute loss for each attention branch. A list of length number_of_attention_branches.
        batch_cross_entropy_losses, _, _, _, _ = process_over_branches(
            batch_predictions,
            label,
            task_type,
            number_of_attention_branches,
            func=criterion,
            use_argmax_for_func=False,
        )

        batch_attention_distribution_penalty = distribution_of_attention(batch_attentions, penalty_weight_for_attention_distribution, threshold_of_attention_distribution)

        # Skip the feature coherence penalty entirely when its weight is 0.
        if loss_config.lambda_for_feature_coherence and branch_attention_inputs is not None:
            batch_feature_coherence = feature_coherence_penalty(
                batch_attentions,
                branch_attention_inputs,
                threshold_scale=loss_config.feature_coherence_threshold_scale,
                sigmoid_sharpness=loss_config.feature_coherence_sigmoid_sharpness,
            )
        else:
            batch_feature_coherence = torch.tensor(0.0)

        
        if number_of_attention_branches > 1:

            batch_similarities = similarity_of_attentions(batch_attentions)  # A list of length number_of_attention_branches
            batch_loss = torch.tensor(0.0, requires_grad=True)
            for i in range(number_of_attention_branches):
                batch_loss = batch_loss + (loss_penalty_with_threshold(batch_cross_entropy_losses[i], threshold_of_loss_penalty) + torch.tensor(C_CE, requires_grad=True)) * (batch_similarities[i] + torch.tensor(C_S, requires_grad=True))
            batch_loss = batch_loss + total_variation_regularization(batch_attentions, lambda_for_total_variation_regularization)
            batch_loss = batch_loss + batch_attention_distribution_penalty

        elif number_of_attention_branches == 1:

            batch_similarities = [torch.tensor(0.0, requires_grad=True)]
            batch_loss = batch_cross_entropy_losses[0] + batch_attention_distribution_penalty
  

        if loss_config.lambda_for_feature_coherence:
            batch_loss = batch_loss + torch.tensor(
                loss_config.lambda_for_feature_coherence, requires_grad=True
            ) * batch_feature_coherence

    return batch_loss, batch_cross_entropy_losses, batch_similarities, batch_attention_distribution_penalty, batch_feature_coherence


def compute_validation_loss(
    model_type,
    model,
    val_loader,
    val_data,
    has_pose,
    criterion,
    epoch,
    loss_config,
    trial,
):

    val_x = val_data.x
    val_label = val_data.label
    val_edge_weight = val_data.edge_weight
    val_x_pose = val_data.x_pose
    task_type = loss_config.task_type
    number_of_attention_branches = loss_config.number_of_attention_branches
    capture_attention_inputs = bool(loss_config.lambda_for_feature_coherence)

    def macro_averaged_accuracy_func(prediction_argmax, label_argmax):
        macro_averaged_accuracy = calculate_macro_averaged_accuracy(prediction_argmax, label_argmax)
        return macro_averaged_accuracy

    model.eval()

    with torch.no_grad():
        
        epoch_loss = []
        epoch_cross_entropy_losses = []
        epoch_similarities = []
        epoch_attention_distribution_penalty = []
        epoch_feature_coherence = []
        epoch_macro_averaged_accuracies = []

        for _, batch_data in enumerate(val_loader):

            label, batch_predictions, batch_attentions, branch_attention_inputs = forward_batch(
                model, model_type, batch_data, has_pose, capture_attention_inputs
            )

            # Compute batch_loss
            # Do not use +=. In-place operations can break the computation graph.
            batch_loss, batch_cross_entropy_losses, batch_similarities, batch_attention_distribution_penalty, batch_feature_coherence = compute_batch_loss(
                batch_predictions,
                batch_attentions,
                label,
                criterion,
                loss_config,
                branch_attention_inputs,
            )

            # Logging
            epoch_loss.append(batch_loss.item())  # A list of length batch_index
            epoch_cross_entropy_losses.append([i.item() for i in batch_cross_entropy_losses])  # A list of length batch_index. [(number_of_attention_branches), (number_of_attention_branches), (number_of_attention_branches)...]
            epoch_similarities.append([i.item() for i in batch_similarities])  # A list of length batch_index. [(number_of_attention_branches), (number_of_attention_branches), (number_of_attention_branches)...]
            epoch_attention_distribution_penalty.append(batch_attention_distribution_penalty.item())
            epoch_feature_coherence.append(batch_feature_coherence.item())

            # Calculate macro averaged accuracy using all val data
            if trial:  # for 02_optuna
                batch_macro_averaged_accuracies, _, _, _, _ = process_over_branches(
                    batch_predictions,
                    label,
                    task_type,
                    number_of_attention_branches,
                    func=macro_averaged_accuracy_func,
                    use_argmax_for_func=True,
                )
                epoch_macro_averaged_accuracies.append([i.item() for i in batch_macro_averaged_accuracies])
        
        # Calculate the average across all batches
        # loss
        loss = sum(epoch_loss) / len(epoch_loss)
        global global_validation_loss
        global_validation_loss.append(loss)

        # cross_entropy_losses
        cross_entropy_losses = []
        for i in zip(*epoch_cross_entropy_losses):  # Extract elements located at the same positions from each sublist within a list.
            cross_entropy_loss = sum(i) / len(i)
            cross_entropy_losses.append(cross_entropy_loss)  # A list of length number_of_attention_branches
        global global_validation_cross_entropy_losses
        global_validation_cross_entropy_losses.append(cross_entropy_losses)  # A list of length number_of_epochs．[(number_of_attention_branches), (number_of_attention_branches), (number_of_attention_branches)...]

        # similarities
        similarities = []
        for i in zip(*epoch_similarities):
            similarity = sum(i) / len(i)
            similarities.append(similarity)  # A list of length number_of_attention_branches
        global global_validation_similarities
        global_validation_similarities.append(similarities)  # A list of length number_of_epochs．[(number_of_attention_branches), (number_of_attention_branches), (number_of_attention_branches)...]

        # attention_distribution_penalty
        attention_distribution_penalty = sum(epoch_attention_distribution_penalty) / len(epoch_attention_distribution_penalty)
        feature_coherence = sum(epoch_feature_coherence) / len(epoch_feature_coherence)
        global global_validation_feature_coherence
        global_validation_feature_coherence.append(feature_coherence)
        global global_validation_attention_distribution_penalty
        global_validation_attention_distribution_penalty.append(attention_distribution_penalty)  # A list of length number_of_epochs．[(number_of_attention_branches), (number_of_attention_branches), (number_of_attention_branches)...]

        # macro_averaged_accuracies
        if trial:  # for 02_optuna
            macro_averaged_accuracies = []
            for i in zip(*epoch_macro_averaged_accuracies):
                accuracy = sum(i) / len(i)
                macro_averaged_accuracies.append(accuracy)  # A list of length number_of_attention_branches

        if not trial:  # for 03_training
            if model_type == "single":
                val_predictions, _ = model(val_x, val_x_pose, has_pose)
            if model_type == "multi":
                val_predictions, _ = model(val_x, val_edge_weight, val_x_pose, has_pose)
            
            macro_averaged_accuracies_tensor, _, _, _, _ = process_over_branches(
                val_predictions,
                val_label,
                task_type,
                number_of_attention_branches,
                func=macro_averaged_accuracy_func,
                use_argmax_for_func=True,
            )
            macro_averaged_accuracies = [i.item() for i in macro_averaged_accuracies_tensor]

    # Save
    with open("validation_results.txt", "a") as file:
        if trial:  # for 02_optuna
            file.write(f"trial_number = {trial.number}\n")
        file.write(f"epoch = {epoch}\n")
        file.write(f"loss = {loss}\n")
        file.write(f"cross_entropy_losses = {cross_entropy_losses}\n")
        file.write(f"similarities = {similarities}\n")
        file.write(f"attention_distribution_penalty = {attention_distribution_penalty}\n")
        if loss_config.lambda_for_feature_coherence:
            file.write(f"feature_coherence = {feature_coherence}\n")
        file.write(f"macro_averaged_accuracies = {macro_averaged_accuracies}\n")
   
    print("validation_loss =", loss)
    print("validation_cross_entropy_losses =", cross_entropy_losses)
    print("validation_similarities         =", similarities)
    print("validation_attention_distribution_penalty =", attention_distribution_penalty)
    if loss_config.lambda_for_feature_coherence:
        print("validation_feature_coherence    =", feature_coherence)
    print("macro_averaged_accuracies       =", macro_averaged_accuracies)

    if trial:  # for 02_optuna
        print("####################")

    if not trial:  # for 03_training
        if min(global_validation_loss) == loss:
            # Save
            if os.path.exists("best_epoch.txt"):
                os.remove("best_epoch.txt")
            with open("best_epoch.txt", "a") as file:
                file.write(f"epoch = {epoch}\n")
                file.write(f"loss = {loss}\n")
                file.write(f"cross_entropy_losses = {cross_entropy_losses}\n")
                file.write(f"similarities = {similarities}\n")
                file.write(f"attention_distribution_penalty = {attention_distribution_penalty}\n")
                if loss_config.lambda_for_feature_coherence:
                    file.write(f"feature_coherence = {feature_coherence}\n")
                file.write(f"macro_averaged_accuracies = {macro_averaged_accuracies}\n")

    return loss, cross_entropy_losses, similarities, attention_distribution_penalty, macro_averaged_accuracies


def train_and_validate(
    model_type,
    model,
    train_loader,
    val_loader,
    val_data,
    has_pose,
    criterion,
    optimizer,
    start_epoch,
    number_of_epochs,
    loss_config,
    trial,  # for 02_optuna
    save_interval_of_epochs,  # for 03_training
):

    task_type = loss_config.task_type
    number_of_attention_branches = loss_config.number_of_attention_branches
    capture_attention_inputs = bool(loss_config.lambda_for_feature_coherence)

    global_step = 0

    for epoch in range(start_epoch, number_of_epochs):

        model.train()  # Set the model to train mode
        
        for _, batch_data in enumerate(train_loader):

            optimizer.zero_grad()
            label, batch_predictions, batch_attentions, branch_attention_inputs = forward_batch(
                model, model_type, batch_data, has_pose, capture_attention_inputs
            )

            # Compute batch_loss
            # Do not use +=. In-place operations can break the computation graph.
            batch_loss, batch_cross_entropy_losses, batch_similarities, batch_attention_distribution_penalty, batch_feature_coherence = compute_batch_loss(
                batch_predictions,
                batch_attentions,
                label,
                criterion,
                loss_config,
                branch_attention_inputs,
            )

            batch_loss.backward()

            optimizer.step()

            # Logging
            global global_training_loss
            global_training_loss.append(batch_loss.item())
            global global_training_cross_entropy_losses
            global_training_cross_entropy_losses.append([i.item() for i in batch_cross_entropy_losses])
            global global_training_similarities
            global_training_similarities.append([i.item() for i in batch_similarities])
            global global_training_feature_coherence
            global_training_feature_coherence.append(batch_feature_coherence.item())
            global global_training_attention_distribution_penalty
            global_training_attention_distribution_penalty.append(batch_attention_distribution_penalty.item())

            if not trial:  # for 03_training
                # Save
                with open("training_results.txt", "a") as file:
                    file.write(f"global_step = {global_step}\n")
                    file.write(f"loss = {global_training_loss[-1]}\n")
                    file.write(f"cross_entropy_losses = {global_training_cross_entropy_losses[-1]}\n")
                    file.write(f"similarities = {global_training_similarities[-1]}\n")
                    file.write(f"attention_distribution_penalty = {global_training_attention_distribution_penalty[-1]}\n")
                    if loss_config.lambda_for_feature_coherence:
                        file.write(f"feature_coherence = {global_training_feature_coherence[-1]}\n")

            global_step += 1

        if trial:  # for 02_optuna
            print("trial_number =", trial.number)
            
        print("epoch =", epoch)
        print("global_step =", global_step)
        print("training_loss =", global_training_loss[-1])
        print("training_cross_entropy_losses   =", global_training_cross_entropy_losses[-1])
        print("training_similarities           =", global_training_similarities[-1])
        print("training_attention_distribution_penalty   =", global_training_attention_distribution_penalty[-1])
        if loss_config.lambda_for_feature_coherence:
            print("training_feature_coherence      =", global_training_feature_coherence[-1])

        if trial:  # for 02_optuna
            # Save
            with open("training_results.txt", "a") as file:
                file.write(f"global_step = {global_step}\n")
                file.write(f"loss = {global_training_loss[-1]}\n")
                file.write(f"cross_entropy_losses = {global_training_cross_entropy_losses[-1]}\n")
                file.write(f"similarities = {global_training_similarities[-1]}\n")
                file.write(f"attention_distribution_penalty = {global_training_attention_distribution_penalty[-1]}\n")
                if loss_config.lambda_for_feature_coherence:
                    file.write(f"feature_coherence = {global_training_feature_coherence[-1]}\n")

        # Compute validation loss
        val_loss, val_cross_entropy_losses, val_similarities, val_attention_distribution_penalty, val_macro_averaged_accuracies = compute_validation_loss(
            model_type,
            model,
            val_loader,
            val_data,
            has_pose,
            criterion,
            epoch,
            loss_config,
            trial,
        )

        # Save the parameter
        if not trial:  # for 03_training
            if save_interval_of_epochs == "best":
                if min(global_validation_loss) == val_loss:
                    # Delete the .pth file of the previous best epoch
                    pth_files = glob.glob("*.pth")
                    if not pth_files:
                        pass
                    else:
                        for pth_file in pth_files:
                            os.remove(pth_file)
                    # Save
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }, f"{epoch}.pth")
                    print(f"########## epoch {epoch} saved ##########")
            elif type(save_interval_of_epochs) == int:
                if epoch % save_interval_of_epochs == (save_interval_of_epochs - 1):
                    # Save
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                    }, f"{epoch}.pth")
                    print(f"########## epoch {epoch} saved ##########")
            # Save to resume from last run
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, "latest_epoch.pth")

    return val_loss, epoch, val_cross_entropy_losses, val_similarities, val_attention_distribution_penalty, val_macro_averaged_accuracies