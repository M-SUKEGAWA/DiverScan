from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import math
from itertools import combinations


forbidden_headers = [
    "Speed",
    "Acceleration",
    "Azimuth",
    "Turning angle",
    "Distance from initial location",
    "Angle from initial location",
    "Distance from userdefined location",
    "Angle from userdefined location",
    "Cumulative travel distance",
    "Cumulative change in turning angle",
]

def to_odd_int(value: float) -> int:
    """
    Convert float to an odd int by rounding.
    If the result is even, add 1 to make it odd.
    """
    val_int = round(value)
    if val_int % 2 == 0:
        val_int += 1
    return val_int

def read_entire_file_lines(csv_path: Path) -> List[str]:
    """
    Read the entire CSV file as a list of raw text lines (stripped of newline).
    """
    with csv_path.open("r", encoding="utf-8-sig") as f:
        return [line.rstrip('\n') for line in f]

def get_total_lines(csv_path: Path) -> int:
    """
    Return the total number of lines (including all header lines and data rows).
    """
    return len(read_entire_file_lines(csv_path))

def get_skeleton_raw_header(lines: List[str]) -> List[str]:
    """
    For skeleton CSV files with 3 header lines.
    Return those 3 lines as a list of strings.
    """
    if len(lines) < 3:
        raise ValueError("Skeleton file must have at least 3 header lines.")
    return lines[1:3]

def get_userset_raw_header(lines: List[str], data_type: str) -> List[str]:
    """
    For userset CSV files:
      - 'single': 1 header line
      - 'multi_animals'/'multi_bodyparts': 2 header lines
    """
    if data_type == "single":
        if len(lines) < 1:
            raise ValueError("Userset 'single' file must have at least 1 header line.")
        return lines[:1]
    elif data_type in ["multi_animals", "multi_bodyparts"]:
        if len(lines) < 2:
            raise ValueError("Userset 'multi' file must have at least 2 header lines.")
        return lines[:2]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def _check_no_nan(df: pd.DataFrame, csv_path: Path) -> None:
    """
    Check NaN in the DataFrame.
    """
    if df.isnull().any().any():
        nan_cols = df.columns[df.isnull().any()].tolist()
        raise ValueError(
            f"Error: '{csv_path.name}' contains NaN values in columns: {nan_cols}"
        )

def read_userset_as_dataframe(csv_path: Path, data_type: str) -> pd.DataFrame:
    """
    For userset CSV:
      - 'single': the 1st line (index=0) is the header → skiprows=[], header=0.
      - 'multi': the 2nd line (index=1) is the header → skiprows=[0], header=0.
    """
    if data_type == "single":
        df = pd.read_csv(csv_path, skiprows=[], header=0, encoding='utf-8-sig')
    elif data_type in ["multi_animals", "multi_bodyparts"]:
        df = pd.read_csv(csv_path, skiprows=[0], header=0, encoding='utf-8-sig')
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    _check_no_nan(df, csv_path)
    return df

def validate_folders(
    class_name_list,
    data_type,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
    number_of_animals,
    number_of_bodyparts,
):
    """Validate every CSV file under the class folders.

    Checks that skeleton and userset files line up, that the required columns
    are present in the expected numbers, and that no forbidden header names or
    NaN values are used. All folders must agree on the moving window size,
    which is returned so that the caller can use it downstream.
    """
    # Dictionary to store moving_window_size per folder
    moving_window_size_dict = {}

    for folder in class_name_list:
        folder_path = Path(folder)
        skeleton_path = folder_path / "skeleton_data_folder"
        userset_path = folder_path / "userset_data_folder"

        # (A) If both skeleton & userset, check filenames and line counts
        if has_both_of_skeleton_and_userset:
            skeleton_files = {f.name for f in skeleton_path.glob("*.csv")}
            userset_files = {f.name for f in userset_path.glob("*.csv")}

            if skeleton_files != userset_files:
                missing_from_skel = userset_files - skeleton_files
                missing_from_user = skeleton_files - userset_files
                msg = f"Error in {folder}:"
                if missing_from_skel:
                    msg += f" Missing in skeleton_data_folder: {missing_from_skel}."
                if missing_from_user:
                    msg += f" Missing in userset_data_folder: {missing_from_user}."
                raise Exception(msg)

            # Check line count conditions
            for csv_name in skeleton_files:
                skel_csv = skeleton_path / csv_name
                user_csv = userset_path / csv_name
                skel_len = get_total_lines(skel_csv)
                user_len = get_total_lines(user_csv)

                if data_type == "single":
                    # (skeleton_len - 3) == (userset_len - 1)
                    if (skel_len - 3) != (user_len - 1):
                        raise Exception(
                            f"Error in {folder}: '{csv_name}' mismatch for data_type='single'. "
                            f"(skeleton_len - 3) != (userset_len - 1): {skel_len}, {user_len}"
                        )
                elif data_type in ["multi_animals", "multi_bodyparts"]:
                    # (skeleton_len - 3) == (userset_len - 2)
                    if (skel_len - 3) != (user_len - 2):
                        raise Exception(
                            f"Error in {folder}: '{csv_name}' mismatch for data_type='{data_type}'. "
                            f"(skeleton_len - 3) != (userset_len - 2): {skel_len}, {user_len}"
                        )
                else:
                    raise Exception(f"Unknown data_type: {data_type}")

        # (B) Calculate moving_window_size
        if (has_both_of_skeleton_and_userset or has_only_skeleton):
            # Use the shortest skeleton CSV
            skeleton_csv_files = list(skeleton_path.glob("*.csv"))
            if not skeleton_csv_files:
                raise Exception(f"Error in {folder}: No skeleton CSV files found.")

            min_skel_len = float('inf')
            for csv_file in skeleton_csv_files:
                length_ = get_total_lines(csv_file)
                if length_ < min_skel_len:
                    min_skel_len = length_

            moving_window_size = to_odd_int((min_skel_len - 3) * 0.03)
            moving_window_size_dict[folder] = moving_window_size

        elif has_only_userset:
            # Use the shortest userset CSV
            userset_csv_files = list(userset_path.glob("*.csv"))
            if not userset_csv_files:
                raise Exception(f"Error in {folder}: No userset CSV files found.")

            min_user_len = float('inf')
            for csv_file in userset_csv_files:
                length_ = get_total_lines(csv_file)
                if length_ < min_user_len:
                    min_user_len = length_

            moving_window_size = to_odd_int((min_user_len - 3) * 0.03)
            moving_window_size_dict[folder] = moving_window_size

        # (C) Check skeleton header consistency if applicable
        if (has_both_of_skeleton_and_userset or has_only_skeleton):
            skeleton_csv_files = list(skeleton_path.glob("*.csv"))
            if skeleton_csv_files:
                base_lines = read_entire_file_lines(skeleton_csv_files[0])
                base_header_3 = get_skeleton_raw_header(base_lines)

                for csv_file in skeleton_csv_files[1:]:
                    curr_lines = read_entire_file_lines(csv_file)
                    curr_header_3 = get_skeleton_raw_header(curr_lines)
                    if curr_header_3 != base_header_3:
                        raise Exception(
                            f"Error in {folder}: skeleton CSV headers (3 lines) differ.\n"
                            f"Base: {skeleton_csv_files[0].name}\n"
                            f"Inconsistent: {csv_file.name}"
                        )

        # (D) Check userset header consistency if applicable
        if (has_both_of_skeleton_and_userset or has_only_userset):
            userset_csv_files = list(userset_path.glob("*.csv"))
            if userset_csv_files:
                base_lines = read_entire_file_lines(userset_csv_files[0])
                base_header = get_userset_raw_header(base_lines, data_type)

                for csv_file in userset_csv_files[1:]:
                    curr_lines = read_entire_file_lines(csv_file)
                    curr_header = get_userset_raw_header(curr_lines, data_type)
                    if curr_header != base_header:
                        raise Exception(
                            f"Error in {folder}: userset CSV headers differ.\n"
                            f"Base: {userset_csv_files[0].name}\n"
                            f"Inconsistent: {csv_file.name}"
                        )

        # (E) Check forbidden columns in userset CSV if applicable
        if (has_both_of_skeleton_and_userset or has_only_userset):
            userset_csv_files = list(userset_path.glob("*.csv"))
            for csv_file in userset_csv_files:
                df_userset = read_userset_as_dataframe(csv_file, data_type)
                for col in df_userset.columns:
                    if col in forbidden_headers:
                        raise Exception(
                            f"Error in {folder}: '{csv_file.name}' has forbidden column '{col}'."
                        )

        # (F) If has_only_userset, check required columns
        if has_only_userset:

            userset_csv_files = list(userset_path.glob("*.csv"))
            for csv_file in userset_csv_files:
                df_userset = read_userset_as_dataframe(csv_file, data_type)
                if data_type == "single":
                    required_cols = {"time", "x", "y"}
                    if not required_cols.issubset(df_userset.columns):
                        raise Exception(
                            f"Error in {folder}: '{csv_file.name}' must contain {required_cols}, "
                            f"but got {set(df_userset.columns)}."
                        )
                if data_type == "multi_animals":
                    # Base names of required columns and their expected counts
                    required_counts = {
                        "time": 1,
                        "x": number_of_animals,
                        "y": number_of_animals,
                    }

                    columns = list(df_userset.columns)

                    for base_name, required in required_counts.items():
                        if base_name in ("x", "y"):
                            # If base_name is 'x' or 'y', count columns named 'x', 'x.1', 'x.2', etc.
                            actual = sum(
                                1
                                for col in columns
                                if col == base_name or col.startswith(f"{base_name}.")
                            )
                        else:
                            # For columns like 'time' that do not have duplicates, count them directly
                            actual = columns.count(base_name)
                        if actual != required:
                            raise Exception(
                                f"Error in {folder}: '{csv_file.name}' Column '{base_name}' count is {actual}, expected {required}."
                            )

                if data_type == "multi_bodyparts":
                    # Base names of required columns and their expected counts
                    required_counts = {
                        "time": 1,
                        "x": number_of_bodyparts,
                        "y": number_of_bodyparts,
                    }

                    columns = list(df_userset.columns)

                    for base_name, required in required_counts.items():
                        if base_name in ("x", "y"):
                            # If base_name is 'x' or 'y', count columns named 'x', 'x.1', 'x.2', etc.
                            actual = sum(
                                1
                                for col in columns
                                if col == base_name or col.startswith(f"{base_name}.")
                            )
                        else:
                            # For columns like 'time' that do not have duplicates, count them directly
                            actual = columns.count(base_name)
                        if actual != required:
                            raise Exception(
                                f"Error in {folder}: '{csv_file.name}' Column '{base_name}' count is {actual}, expected {required}."
                            )

    print("All checks passed.")
    print("moving_window_size_dict:", moving_window_size_dict)

    return moving_window_size_dict[list(moving_window_size_dict.keys())[0]]  # All should be a same value


def validate_input_values(
    data_type,
    number_of_animals,
    number_of_bodyparts,
    location_of_animal,
    bodyparts_to_calculate_centroid,
    head_bodypart,
    tail_bodypart,
    head_direction,
    userdefined_features,
    names_of_userdefined_features,
    graph_type,
    userset_edges,
    head_direction_difference_between_nodes,
    bodypart_to_bodypart_distance_between_nodes,
    bodyparts_pair_list,
):
    """Validate the user-supplied settings and normalise them.

    Raises ValueError with an explanatory message when a setting is
    inconsistent with the chosen data type. Returns the canonical data type
    ("single", "multi_animals" or "multi_bodyparts") together with the
    validated bodyparts_list_for_pose.
    """
    if data_type == "single (set number_of_animals = 1)":
        data_type = "single"  # Name data_type
        if not (number_of_animals == 1):
            raise ValueError('Check "number_of_animals". Set 1 if data_type = single.')
    if data_type == "multi-animal (set number_of_animals >= 2)":
        data_type = "multi_animals"  # Name data_type
        if not (number_of_animals >= 2):
            raise ValueError('Check "number_of_animals". Set >= 2 if data_type = multi-animal.')
    if data_type == "multi-bodypart (set number_of_animals = 1 and number_of_bodyparts >= 2)":
        data_type = "multi_bodyparts"  # Name data_type
        if not (number_of_animals == 1):
            raise ValueError('Check "number_of_animals". Set 1 if data_type = multi-bodypart.')

    if data_type == "multi_bodyparts" and number_of_bodyparts < 2:
        raise ValueError('Check "number_of_bodyparts". Set >= 2 if data_type = multi-bodypart.')
    if number_of_bodyparts < 1:
        raise ValueError('Check "number_of_bodyparts". Set a natural number. If you do not have "skeleton_data_folder", set 1 as dummy data.')

    if data_type == "single" or data_type == "multi_animals":
        if location_of_animal == "Use x and y columns in userset data (set bodyparts_to_calculate_centroid = None)":
            if bodyparts_to_calculate_centroid is not None:
                raise ValueError('Check "bodyparts_to_calculate_centroid". Set None if location_of_animal == "Use x and y".')
        if location_of_animal == "Calculate centroid(s) of animal(s) using skeleton data (set the bodyparts_to_calculate_centroid)":
            if not isinstance(bodyparts_to_calculate_centroid, list):
                raise ValueError('Check the input format of "bodyparts_to_calculate_centroid". Use the list format if location_of_animal = "Calculate centroid(s) of animal(s) using skeleton data". eg., [1] / [1, 2, 3]')
            if not all(isinstance(item, int) for item in bodyparts_to_calculate_centroid):
                raise ValueError('Check the input format of "bodyparts_to_calculate_centroid". Use only natural number in your list, if location_of_animal = "Calculate centroid(s) of animal(s) using skeleton data". eg., [1] / [1, 2, 3]')
            if any(item > number_of_bodyparts for item in bodyparts_to_calculate_centroid) or any(item <= 0 for item in bodyparts_to_calculate_centroid):
                raise ValueError('Check the input format of "bodyparts_to_calculate_centroid". Numbers in the list should be in the range from 1 to number_of_bodyparts, if location_of_animal = "Calculate centroid(s) of animal(s) using skeleton data". eg., [1] / [1, 2, 3]')
        if location_of_animal == "data_type is 'multi-bodypart' (set bodyparts_to_calculate_centroid = None)":
            raise ValueError('Set "Calculate centroid(s)" or "Use x and y", if data_type = single or data_type = multi-animal.')
    if data_type == "multi_bodyparts":
        if not location_of_animal == "data_type is 'multi-bodypart' (set bodyparts_to_calculate_centroid = None)":
            raise ValueError('Check "location_of_animal". Set "Do not use this parameter" if data_type = multi-bodypart.')
        if bodyparts_to_calculate_centroid is not None:
            raise ValueError('Check "bodyparts_to_calculate_centroid". Set None if location_of_animal == "Do not use this parameter".')

    if number_of_bodyparts == 1 or data_type == "multi_bodyparts":
        if not (head_bodypart == 0) or not (tail_bodypart == 0):
            raise ValueError('Check "head_bodypart" and "tail_bodypart". Set 0 for both if number_of_bodyparts = 1 or data_type = multi-bodypart.')
    if data_type == "single" or data_type == "multi_animals":
        if head_bodypart == 0 and tail_bodypart == 0:
            pass  # Not calculate head_direction
        else:
            if head_bodypart == tail_bodypart:
                raise ValueError('Check the input of "head_bodypart". Do not set the same bodypart for "head_bodypart" and "tail_bodypart", except the case of both are 0.')
            if (head_bodypart > number_of_bodyparts) or (head_bodypart <= 0):
                raise ValueError('Check the input of "head_bodypart". The value should be in the range from 1 to number_of_bodyparts, except the case both of "head_bodypart" and "tail_bodypart" are 0.')
            if (tail_bodypart > number_of_bodyparts) or (tail_bodypart <= 0):
                raise ValueError('Check the input of "tail_bodypart". The value should be in the range from 1 to number_of_bodyparts, except the case both of "head_bodypart" and "tail_bodypart" are 0.')


    bodyparts_list_for_pose = "Default"
    if not(((data_type == "single") and (number_of_bodyparts >= 3)) or ((data_type == "multi_animals") and (number_of_bodyparts >= 3))):
    	bodyparts_list_for_pose = None

    # if isinstance(bodyparts_list_for_pose, list):
    #     bodyparts_list_for_pose = np.array(bodyparts_list_for_pose) - 1  # (number of 3 points, 3). Minus 1 from the array to start the bodypart number from zero.
    # if ((data_type == "single") and (number_of_bodyparts >= 3)) or ((data_type == "multi_animals") and (number_of_bodyparts >= 3)):
    #     if setting_for_pose == "Default calculation (set bodyparts_list_for_pose = 'Default')":
    #         if not(bodyparts_list_for_pose == "Default"):
    #             raise ValueError('Check "bodyparts_list_for_pose". Set "Default" if setting_for_pose = "Default".')
    #     if setting_for_pose == "Angle of three bodyparts (set the bodyparts_list_for_pose)":
    #         if not isinstance(bodyparts_list_for_pose, list):
    #             raise ValueError('Check the input format of "bodyparts_list_for_pose". Use the nested-list format if setting_for_pose = "Angle of three bodyparts". eg., [[1, 2, 3], [1, 2, 4], [2, 3, 4]] / [[1, 2, 3]]')
    #         for item in bodyparts_list_for_pose:
    #             if not isinstance(item, list):
    #                 raise ValueError('Check the input format of "bodyparts_list_for_pose". Use the nested-list format if setting_for_pose = "Angle of three bodyparts". eg., [[1, 2, 3], [1, 2, 4], [2, 3, 4]] / [[1, 2, 3]]')
    #             if not len(item) == 3:
    #                 raise ValueError('Check the input format of "bodyparts_list_for_pose". The length of each sublist should be 3, if setting_for_pose = "Angle of three bodyparts". eg., [[1, 2, 3], [1, 2, 4], [2, 3, 4]] / [[1, 2, 3]]')
    #             for sub_item in item:
    #                 if isinstance(sub_item, list):
    #                     raise ValueError('Check the input format of "bodyparts_list_for_pose". The list is nested too deeply: the depth should be 2, if setting_for_pose = "Angle of three bodyparts. eg., [[1, 2, 3], [1, 2, 4], [2, 3, 4]] / [[1, 2, 3]]')
    #                 if not isinstance(sub_item, int):
    #                     raise ValueError('Check the input format of "bodyparts_list_for_pose". Use only natural number in your list if setting_for_pose = "Angle of three bodyparts".')
    #                 if sub_item > number_of_bodyparts or sub_item <= 0:
    #                     raise ValueError('Check the input format of "bodyparts_list_for_pose". Numbers in the list should be in the range from 1 to number_of_bodyparts, if setting_for_pose = "Angle of three bodyparts".')

    ##### Features for machine learning #####

    if head_bodypart == 0 and tail_bodypart == 0:
        if head_direction == True:
            raise ValueError('Check "head_direction". Do not check it if head_bodypart = 0 and tail_bodypart = 0.')

    if userdefined_features == "No (set names_of_userdefined_features = None)":
        if names_of_userdefined_features is not None:
            raise ValueError('Check "names_of_userdefined_features". Set None if userdefined_features = "No".')
    if userdefined_features == "Yes (set exact column names in the names_of_userdefined_features field)":
        if not isinstance(names_of_userdefined_features, list):
            raise ValueError('Check the input format of "names_of_userdefined_features". Use the list format if userdefined_features = "Yes". eg., ["height"] / ["latitude", "longitude"]')

    ##### Settings for multi #####

    if data_type == "multi_animals" or data_type == "multi_bodyparts":

        if graph_type == "complete graph (set userset_edges = None)":
            if userset_edges is not None:
                 raise ValueError('Check "userset_edges". Set None if graph_type = "complete graph.')
        if graph_type == "userset_edges (set your userset_edges)":
            if not isinstance(userset_edges, list):
                raise ValueError('Check the input format of "userset_edges". Use the nested-list format if graph_type = "userset_edges". eg., [[1, 2], [2, 1]] / [[1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [4, 1], [1, 5], [5, 1]]')
            for item in userset_edges:
                if not isinstance(item, list):
                    raise ValueError('Check the input format of "userset_edges". Use the nested-list format if graph_type = "userset_edges". eg., [[1, 2], [2, 1]] / [[1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [4, 1], [1, 5], [5, 1]]')
                if not len(item) == 2:
                    raise ValueError('Check the input format of "userset_edges". The length of each sublist should be 2, if graph_type = "userset_edges". eg., [[1, 2], [2, 1]] / [[1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [4, 1], [1, 5], [5, 1]]')
                for sub_item in item:
                    if isinstance(sub_item, list):
                        raise ValueError('Check the input format of "userset_edges". The list is nested too deeply: the depth should be 2, if graph_type = "userset_edges". eg., [[1, 2], [2, 1]] / [[1, 2], [2, 1], [1, 3], [3, 1], [1, 4], [4, 1], [1, 5], [5, 1]]')
                    if not isinstance(sub_item, int):
                        raise ValueError('Check the input format of "userset_edges". Use only natural number in your list if graph_type = "userset_edges".')
                    if data_type == "multi_animals":
                        if sub_item > number_of_animals or sub_item <= 0:
                            raise ValueError('Check the input format of "userset_edges". Numbers in the list should be in the range from 1 to number_of_animals if graph_type = "userset_edges" and data_type = "multi-animal".')
                    if data_type == "multi_bodyparts":
                        if sub_item > number_of_bodyparts or sub_item <= 0:
                            raise ValueError('Check the input format of "userset_edges". Numbers in the list should be in the range from 1 to number_of_bodyparts if graph_type = "userset_edges" and data_type = "multi-bodypart".')

        if (head_bodypart == 0 and tail_bodypart == 0):
            if head_direction_difference_between_nodes == True:
                raise ValueError('Check "head_direction_difference_between_nodes". Do not check it if head_bodypart = 0 and tail_bodypart = 0.')

        if (data_type == "multi_animals") and (number_of_bodyparts >= 2):
            if bodypart_to_bodypart_distance_between_nodes == "No (set bodyparts_pair_list = None)":
                if bodyparts_pair_list is not None:
                    raise ValueError('Check "bodyparts_pair_list". Set None if bodypart_to_bodypart_distance_between_nodes = "No"')
            if bodypart_to_bodypart_distance_between_nodes == "Yes (set the bodyparts_pair_list)":
                if not isinstance(bodyparts_pair_list, list):
                    raise ValueError('Check the input format of "bodyparts_pair_list". Use the nested-list format if bodypart_to_bodypart_distance_between_nodes = "Yes". eg., [[1, 1], [1, 5]] / [[1, 2]]')
                for item in bodyparts_pair_list:
                    if not isinstance(item, list):
                        raise ValueError('Check the input format of "bodyparts_pair_list". Use the nested-list format if bodypart_to_bodypart_distance_between_nodes = "Yes". eg., [[1, 1], [1, 5]] / [[1, 2]]')
                    if not len(item) == 2:
                        raise ValueError('Check the input format of "bodyparts_pair_list". The length of each sublist should be 2, if bodypart_to_bodypart_distance_between_nodes = "Yes". eg., [[1, 1], [1, 5]] / [[1, 2]]')
                    for sub_item in item:
                        if isinstance(sub_item, list):
                            raise ValueError('Check the input format of "bodyparts_pair_list". The list is nested too deeply: the depth should be 2, if bodypart_to_bodypart_distance_between_nodes = "Yes. eg., [[1, 1], [1, 5]] / [[1, 2]]')
                        if not isinstance(sub_item, int):
                            raise ValueError('Check the input format of "bodyparts_pair_list". Use only natural number in your list if bodypart_to_bodypart_distance_between_nodes = "Yes".')
                        if sub_item > number_of_bodyparts or sub_item <= 0:
                            raise ValueError('Check the input format of "bodyparts_pair_list". Numbers in the list should be in the range from 1 to number_of_bodyparts, if bodypart_to_bodypart_distance_between_nodes = "Yes".')
        else:
            if bodypart_to_bodypart_distance_between_nodes == "Yes (set the bodyparts_pair_list)":
                raise ValueError('Check "bodypart_to_bodypart_distance_between_nodes". Do not select "None" if not multi-animal with number_of_bodyparts >= 2.')
            if bodyparts_pair_list is not None:
                raise ValueError('Check "bodyparts_pair_list". Set None if bodypart_to_bodypart_distance_between_nodes = "No"')
    ##########

    return data_type, bodyparts_list_for_pose


def build_feature_name_lists(
    data_type,
    time,
    x,
    y,
    speed,
    azimuth,
    turning_angle,
    acceleration,
    distance_from_userdefined_location,
    angle_from_userdefined_location,
    distance_from_initial_location,
    angle_from_initial_location,
    cumulative_travel_distance,
    cumulative_change_in_turning_angle,
    head_direction,
    names_of_userdefined_features,
    distance_between_nodes,
    azimuth_difference_between_nodes,
    head_direction_difference_between_nodes,
    bodyparts_pair_list,
):
    """Build the feature name lists used for machine learning.

    Returns the per-node feature names and, for the multi models, the edge
    feature names. Raises ValueError when no feature is selected.
    """
    input_feature_dict = {
        "time": time,
        "x": x,
        "y": y,
        "Speed": speed,
        "Azimuth": azimuth,
        "Turning angle": turning_angle,
        "Acceleration": acceleration,
        "Distance from userdefined location": distance_from_userdefined_location,
        "Angle from userdefined location": angle_from_userdefined_location,
        "Distance from initial location": distance_from_initial_location,
        "Angle from initial location": angle_from_initial_location,
        "Cumulative travel distance": cumulative_travel_distance,
        "Cumulative change in turning angle": cumulative_change_in_turning_angle,
        "Head direction": head_direction,
    }

    if names_of_userdefined_features is not None:
        for feature in names_of_userdefined_features:
            input_feature_dict[feature] = True

    feature_name_list_for_machine_learning = [key for key, item in input_feature_dict.items() if item is not None and item is not False]  # To be used later

    if feature_name_list_for_machine_learning == []:
        raise ValueError('Set at least one feature for machine learning.')

    ##########

    if data_type == "multi_animals" or data_type == "multi_bodyparts":

        input_edge_data_dict = {
            "Distance between nodes": distance_between_nodes,
            "Azimuth difference between nodes": azimuth_difference_between_nodes,
            "Head direction difference between nodes": head_direction_difference_between_nodes,
            "Bodypart-to-bodypart distance between nodes": bodyparts_pair_list,
        }

        edge_data_name_list_for_machine_learning = []  # To be used later
        for key, item in input_edge_data_dict.items():
            if item is not None and item is not False:
                if key == "Bodypart-to-bodypart distance between nodes":
                    # Dynamically generate names from bodyparts_pair_list
                    for pair in item:
                        bodypart_1, bodypart_2 = pair
                        dynamic_name = f"Bodypart{bodypart_1}-to-bodypart{bodypart_2} distance between nodes"
                        edge_data_name_list_for_machine_learning.append(dynamic_name)
                else:
                    edge_data_name_list_for_machine_learning.append(key)

        if edge_data_name_list_for_machine_learning == []:
            raise ValueError('Set at least one edge attribute for machine learning.')
    else:
        edge_data_name_list_for_machine_learning = None

    return feature_name_list_for_machine_learning, edge_data_name_list_for_machine_learning


def read_csv_files_and_make_onehot_labels(
    class_name_list,
    data_type,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
):
    """
    Read the csv files and create dataset.
    """


    # Initialize the new dictionary
    data_category_names = ["skeleton_data", "userset_data", "file_names", "ohehot_labels"]

    classes_dictionary = {}
    for class_name in class_name_list:
        classes_dictionary[class_name] = {}
        for data_category in data_category_names:
            classes_dictionary[class_name][data_category] = []

    # Prepare one-hot label
    one_hot = np.eye(len(class_name_list))
    
    # Read csv files
    for i, class_name in enumerate(class_name_list):
        
        if has_both_of_skeleton_and_userset or has_only_skeleton:
            # skeleton
            subdir = Path(class_name) / "skeleton_data_folder"
            csv_files = sorted(subdir.glob("*.csv"))

            skeleton_data = []
            file_names = []
            onehot_labels = []            
            for csv_file in csv_files:
                data = pd.read_csv(csv_file, header=[0, 1, 2])
                file_name = str(Path(class_name) / csv_file.name)
                skeleton_data.append(data)
                file_names.append(file_name)
                onehot_labels.append(one_hot[i])

            classes_dictionary[class_name]["skeleton_data"] = skeleton_data
            classes_dictionary[class_name]["file_names"] = file_names
            classes_dictionary[class_name]["onehot_labels"] = onehot_labels

        if has_both_of_skeleton_and_userset or has_only_userset:
            # userset
            subdir = Path(class_name) / "userset_data_folder"
            csv_files = sorted(subdir.glob("*.csv"))

            userset_data = []
            if has_only_userset:
                file_names = []
                onehot_labels = []

            for csv_file in csv_files:
                if data_type == "single":
                    data = pd.read_csv(csv_file, header=[0])
                if data_type == "multi_animals" or data_type == "multi_bodyparts":
                    data = pd.read_csv(csv_file, header=[0, 1])
                userset_data.append(data)

                if has_only_userset:
                    file_name = str(Path(class_name) / csv_file.name)
                    file_names.append(file_name)
                    onehot_labels.append(one_hot[i])
                    
            classes_dictionary[class_name]["userset_data"] = userset_data
            if has_only_userset:
                classes_dictionary[class_name]["file_names"] = file_names
                classes_dictionary[class_name]["onehot_labels"] = onehot_labels
                 
    return classes_dictionary


def create_train_val_test_set(
    classes_dictionary,
    class_name_list,
    set_names,
    train_ratio,
    val_ratio,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
    split_seed=0,
    data_category_names=None,
):
    """Split every class into train, val and test sets.

    The split is drawn from a generator seeded with split_seed, so the same
    seed always produces the same split regardless of what else has drawn
    from NumPy's global random state.

    data_category_names overrides which entries of classes_dictionary are
    carried through the split, for callers that hold something other than
    skeleton and userset data.
    """
    rng = np.random.RandomState(split_seed)
    
    log_file = open("dataset_split.txt", "a")
    
    def split_data(data, train_ratio, val_ratio):
        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)
        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
    
    def add_split_data_to_dictionary(class_name, set_names, data_category_names, classes_dictionary, dictionary, permutation, train_ratio, val_ratio):
        for data_category in data_category_names:
            data = classes_dictionary[class_name][data_category]
            permuted_data = [data[i] for i in permutation]
            split = split_data(permuted_data, train_ratio, val_ratio)
            for set_name in set_names:
                dictionary[set_name][data_category].extend(split[set_name])
        return dictionary
            
    
    # Initialize the new dictionary
    all_data_category_names = data_category_names if data_category_names is not None else [

        "skeleton_data", 
        "userset_data", 
        "file_names", 
        "onehot_labels",
        "skeleton_data_formatted",
        "userset_data_formatted",
        "skeleton_data_corrected",
        "time_x_y_data",
        "skeleton_nan_padded_array",
        "userset_nan_padded_array",
        "time_x_y_nan_padded_array",
        "onehot_labels_array",
        "features",
        "pose_feature",
        "edge_index",
        "edge_data",
        "features_for_machine_learning",
        "edge_data_for_machine_learning",
        "features_standardized",
        "edge_data_standardized",
        "pose_feature_standardized",
        "edge_attributes",
        "features_for_post_analysis",
    ]

    dictionary = {}
    for set_name in set_names:
        dictionary[set_name] = {}
        for data_category in all_data_category_names:
            dictionary[set_name][data_category] = []

    # Split data
    for class_name in class_name_list:

        # Generate a single permutation for all data types in this class
        num_items = len(classes_dictionary[class_name]["file_names"])
        permutation = rng.permutation(num_items)

        if data_category_names is None:
            if has_both_of_skeleton_and_userset:
                data_category_names = ["skeleton_data", "userset_data", "file_names", "onehot_labels"]
            if has_only_skeleton:
                data_category_names = ["skeleton_data", "file_names", "onehot_labels"]
            if has_only_userset:
                data_category_names = ["userset_data", "file_names", "onehot_labels"]

        dictionary = add_split_data_to_dictionary(
            class_name, 
            set_names, 
            data_category_names, 
            classes_dictionary, 
            dictionary, 
            permutation, 
            train_ratio, 
            val_ratio,
        )

    # A set left empty would only surface later as an unrelated IndexError.
    empty_sets = [set_name for set_name in set_names if len(dictionary[set_name]["file_names"]) == 0]
    if empty_sets:
        counts = ", ".join(
            f"{class_name}: {len(classes_dictionary[class_name]['file_names'])}"
            for class_name in class_name_list
        )
        raise ValueError(
            f"The {' and '.join(empty_sets)} set(s) received no files. Each class is split "
            f"on its own with train_ratio={train_ratio} and val_ratio={val_ratio}, and the "
            f"number of files per class ({counts}) is too small for every set to get at "
            f"least one. Add files, or adjust the ratios."
        )

    # Final shuffle for each set, maintaining the alignment across data types
    for set_name in set_names:
        num_items = len(dictionary[set_name]["file_names"])
        final_permutation = rng.permutation(num_items)
        for data_category in data_category_names:
            data = dictionary[set_name][data_category]
            dictionary[set_name][data_category] = [data[i] for i in final_permutation]

        print(f"{set_name}: {num_items} files")
        log_file.write(f"{set_name}: {num_items} files\n")

        for file_name in dictionary[set_name]["file_names"]:
            print(file_name)
            log_file.write(file_name + "\n") 

    log_file.close()

    return dictionary


def format_skeleton_data(
    dataframes, 
    number_of_animals, 
    number_of_bodyparts,
):
    """
    Return:
    data: [[array, array...], [array, array...]...]     (outer list: length of number of classes; inner list: length of number of files of the class)
        array shape:
            single: (1, number_of_bodyparts, number_of_timestamps, 3(x, y, likelihood))
            multi_animals: (number_of_animals, number_of_bodyparts, number_of_timestamps, 3(x, y, likelihood))
            multi_bodyparts: (1, number_of_bodyparts, number_of_timestamps, 3(x, y, likelihood))
    """   

    number_of_cols_per_individual = number_of_bodyparts * 3  # 3 : x, y, likelihood

    data = []
    for dataframe in dataframes:

        number_of_timestamps = dataframe.shape[0]

        array = np.zeros((number_of_animals, number_of_bodyparts, number_of_timestamps, 3))

        for i in range(number_of_animals):
            for j in range(number_of_bodyparts):
                base_col_index = (i * number_of_cols_per_individual) + (j * 3) + 1  # +1 : time col
                
                x_values = dataframe.iloc[:, base_col_index].values
                y_values = dataframe.iloc[:, base_col_index + 1].values
                likelihood_values = dataframe.iloc[:, base_col_index + 2].values
                
                array[i, j, :, 0] = x_values
                array[i, j, :, 1] = y_values
                array[i, j, :, 2] = likelihood_values

        data.append(array)

    return data


def format_userset_data(
    dataframes, 
    data_type, 
    number_of_animals,
    number_of_bodyparts,
):
    """
    Returns:
    data: [[array, array...], [array, array...]...]     (outer list: length of number of classes; inner list: length of number of files of the class)
        array shape:
            single: (1, number_of_timestamps, number_of_cols(eg., time, x, y, latitude, longitude...))
            multi_animals: (number_of_animals, number_of_timestamps, number_of_cols(eg., time, x, y, latitude, longitude...))
    column_name_list: names of the columns in the csv file.
    """

    if data_type == "single":

        data = []
        for dataframe in dataframes:
            array = dataframe.to_numpy()
            column_name_list = dataframe.columns.tolist()  # All data have the same column name
            array = np.expand_dims(array, axis=0)  # (1, time, number_of_cols)
            data.append(array)
        
    if data_type == "multi_animals" or data_type == "multi_bodyparts":

        if data_type == "multi_bodyparts":
            number_of_animals = number_of_bodyparts

        data = []
        for dataframe in dataframes:
            
            # Calculate number of cols per animal (and get time_data)
            if "time" in dataframe.columns:
                time_data = dataframe["time"]["time"].to_numpy()
                number_of_cols_per_animal = (dataframe.shape[1] - 1) // number_of_animals  # -1 : time col
            else:
                number_of_cols_per_animal = dataframe.shape[1] // number_of_animals
            
            # Extract columns corresponding to each animal
            animal_arrays = []
            for i in range(number_of_animals):

                # Select columns for the animal
                if "time" in dataframe.columns:
                    col_indices = [col_index for col_index in range(i*number_of_cols_per_animal+1, (i+1)*number_of_cols_per_animal+1)]
                else:
                    col_indices = [col_index for col_index in range(i*number_of_cols_per_animal, (i+1)*number_of_cols_per_animal)]
                
                # Create a dataframe using the selected columns
                animal_df = dataframe.iloc[:, col_indices].copy()

                # Simplify column names (keeping only the second level of the multi-index)
                animal_df.columns = [col[1] for col in animal_df.columns]
                
                # Add the "time" column
                if "time" in dataframe.columns:
                    animal_df["time"] = time_data
                
                animal_arrays.append(animal_df.to_numpy())
                column_name_list = animal_df.columns.tolist()  # All data have the same column name
            
            array = np.stack(animal_arrays, axis=0)  # Stack to make it (number_of_animals, T, number_of_cols)
            data.append(array)
        
    return data, column_name_list


def format_data(
    dictionary,
    set_names,
    data_type,
    number_of_animals,
    number_of_bodyparts,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
):

    for set_name in set_names:

        if has_both_of_skeleton_and_userset:
            dictionary[set_name]["skeleton_data_formatted"] = format_skeleton_data(
                dictionary[set_name]["skeleton_data"],
                number_of_animals,
                number_of_bodyparts,
            )

            dictionary[set_name]["userset_data_formatted"], column_name_list = format_userset_data(
                dictionary[set_name]["userset_data"],
                data_type,
                number_of_animals,
                number_of_bodyparts,
            )

        if has_only_skeleton:
            dictionary[set_name]["skeleton_data_formatted"] = format_skeleton_data(
                dictionary[set_name]["skeleton_data"],
                number_of_animals,
                number_of_bodyparts,
            )
            dictionary[set_name]["userset_data_formatted"] = None
            column_name_list = None

        if has_only_userset:
            dictionary[set_name]["skeleton_data_formatted"] = None
            dictionary[set_name]["userset_data_formatted"], column_name_list = format_userset_data(
                dictionary[set_name]["userset_data"],
                data_type,
                number_of_animals,
                number_of_bodyparts,
            )

    return dictionary, column_name_list


def replace_low_likelihood_coords(
    arrays,
    file_names,
    threshold_likelihood,
):

    """
    NOW NO CORRECTION IS PERFORMED HERE.

    If the likelihood is below the threshold, 
    the (x, y) coordinates will be masked and then interpolated
    using linear interpolation from neighboring valid coordinates.

    Return:
    skeleton_data: [array, array...]
        array shape:
            single: (1, number_of_bodyparts, number_of_timestamps, 2(x, y))
            multi_animals: (number_of_animals, number_of_bodyparts, number_of_timestamps, 2(x, y))
            multi_bodyparts: (1, number_of_bodyparts, number_of_timestamps, 2(x, y))
    """

    # log_file = open("likelihood_based_corrections.txt", "a")

    skeleton_data = []

    for file_name, array in zip(file_names, arrays):

        array = np.copy(array)  # Deepcopy is necessary to prevent altering the original array
        number_of_animals, number_of_bodyparts, number_of_timestamps, _ = array.shape

        # Initialize counter for corrections
        likelihood_based_correction_counter = 0

        for i in range(number_of_animals):
            for j in range(number_of_bodyparts):
                # Extract x, y, likelihood
                x = array[i, j, :, 0]
                y = array[i, j, :, 1]
                likelihood = array[i, j, :, 2]

                # Create mask where likelihood is below threshold
                # mask = likelihood < threshold_likelihood
                mask = np.full(likelihood.shape, False, dtype=bool)  # NOW NO CORRECTION IS PERFORMED HERE

                # Count the number of corrections
                likelihood_based_correction_counter += np.sum(mask)

                # Set x and y to NaN where masked
                x_masked = x.copy()
                y_masked = y.copy()
                x_masked[mask] = np.nan
                y_masked[mask] = np.nan

                # Interpolate over NaNs
                x_interp = pd.Series(x_masked).interpolate(method='linear', limit_direction='both').to_numpy()
                y_interp = pd.Series(y_masked).interpolate(method='linear', limit_direction='both').to_numpy()

                # Replace the original x and y with interpolated values
                array[i, j, :, 0] = x_interp
                array[i, j, :, 1] = y_interp

        skeleton_data.append(array[:, :, :, 0:2])

        # print(f"File: {file_name}")
        # print(f"Total number of likelihood based corrections: {likelihood_based_correction_counter}")
        # log_file.write(f"File: {file_name}\n")
        # log_file.write(f"Total number of likelihood based corrections: {likelihood_based_correction_counter}\n")

    # log_file.close()

    return skeleton_data


def replace_fast_moving_coords(
    arrays,
    file_names,
    threshold_distance,
    column_name_list,
):

    """
    NOW NO CORRECTION IS PERFORMED HERE.

    If the moving distance calculated by t (x, y) and t-1(x', y') are above the threshold, 
    they will be replaced with the values from the nearest past timestamp that is below the threshold.

    if column_name_list:
        Return:
        data: [array, array...]
            array shape:
                single: (1, 1, number_of_timestamps, 3(time, x, y))
                multi_animals: (number_of_animals, 1, number_of_timestamps, 3(time, x, y))

    else:
        Return:
        data: [array, array...]
            array shape:
                single: (1, number_of_bodyparts, number_of_timestamps, 3(time, x, y))
                multi_animals: (number_of_animals, number_of_bodyparts, number_of_timestamps, 3(time, x, y))
                multi_bodyparts: (1, number_of_bodyparts, number_of_timestamps, 3(time, x, y))
    """

    # log_file = open("distance_based_corrections.txt", "a")

    data = []

    for file_name, array in zip(file_names, arrays):

        if column_name_list:
            try:
                index_of_x = column_name_list.index("x")
                index_of_y = column_name_list.index("y")
                index_of_time = column_name_list.index("time")
            except ValueError as e:
                print('You should include["time", "x", "y"] in the all csv files in the "userset_data_folder". Check', e)

            time_array = array[:, np.newaxis, :, [index_of_time]]  # (number_of_animals, 1, number_of_timestamps, 1(time)). For real value of "time".
            array = array[:, np.newaxis, :, [index_of_x, index_of_y]]  # (number_of_animals, 1, number_of_timestamps, 2(x, y))

        array = np.copy(array)  # deepcopy is necessary to prevent altering the original array
        number_of_animals, number_of_bodyparts, number_of_timestamps, _ = array.shape

        if not column_name_list:
            time_values = np.arange(number_of_timestamps).reshape(1, 1, number_of_timestamps, 1)  # 0, 1, 2...(number_of_timestamps-1)
            time_array = np.broadcast_to(time_values, (number_of_animals, number_of_bodyparts, number_of_timestamps, 1))  # (number_of_animals, 1, number_of_timestamps, 1(time))

        # If the movement distance from the previous timestamp exceeds the specified value, 
        # replace it with the previous value. This process is done row by row to handle cases 
        # where the entity stays at the destination for several timestamps.
        # distance_based_correction_counter = 0
        # for i in range(number_of_animals):
        #     for j in range(number_of_bodyparts):
        #         for k in range(number_of_timestamps):
        #             if k == 0:
        #                 distance = 0
        #             else:
        #                 distance = np.sqrt((array[i, j, k, 0] - array[i, j, k-1, 0])**2 + (array[i, j, k, 1] - array[i, j, k-1, 1])**2)

                    # if distance > threshold_distance:
                    #     # Search for the nearest past value with a distance below the threshold
                    #     # Initialize with the value at timestamp 0
                    #     nearest_past_value = array[i, j, 0, :]
                    #     for l in range(k):
                    #         distance_candidate = np.sqrt((array[i, j, (k-l), 0] - array[i, j, (k-l-1), 0])**2 + (array[i, j, (k-l), 1] - array[i, j, (k-l-1), 1])**2)
                    #         if distance_candidate <= threshold_distance:
                    #             nearest_past_value = array[i, j, (k-l), :]  # If the distance at timestamp k is below the threshold, retain the coordinates at that point as the "nearest past value".
                    #             break  # Break the loop once a suitable value is found
                    #     # Replace the value
                    #     array[i, j, k, :] = nearest_past_value
                    #     distance_based_correction_counter += 1
                    #     # print("Distance based correction.", "animal:", i, "bodypart:", j, "time:", k)

        data.append(np.concatenate((time_array, array), axis=3))

    #     print(f"File: {file_name}")
    #     print(f"Total number of distance based corrections: {distance_based_correction_counter}")
    #     log_file.write(f"File: {file_name}\n")
    #     log_file.write(f"Total number of distance based corrections: {distance_based_correction_counter}\n")

    # log_file.close()

    return data


def get_time_x_y_data(
    arrays,
    data_type,
    bodyparts_list_to_calculate_centroid,
):

    """
    Return:
    data: [array, array...]
        array shape:
            single: (1, number_of_timestamps, 3(time, x, y))
            multi_animals: (number_of_animals, number_of_timestamps, 3(time, x, y))
            multi_bodyparts: (number_of_bodyparts, number_of_timestamps, 3(time, x, y))
    """

    data = []

    for array in arrays:

        if data_type == "single" or data_type == "multi_animals":
            if bodyparts_list_to_calculate_centroid is None:  # Use x, y from userset_data_folder
                array = np.squeeze(array, axis=1)  # axis=1: the dimention of number_of_bodyparts
            else:
                number_of_animals, number_of_bodyparts, number_of_timestamps, _ = array.shape

                centroids = np.mean(array[:, bodyparts_list_to_calculate_centroid, :, 1:], axis=1)
                time_data = array[0, 0, :, 0]  # "time" is same over all animals and bodyparts

                array = np.empty((number_of_animals, number_of_timestamps, 3))
                array[:, :, 0] = time_data
                array[:, :, 1:] = centroids

        if data_type == "multi_bodyparts":
            array = np.squeeze(array, axis=0)  # axis=0: the dimention of number_of_animals
        
        data.append(array)

    return data


def correct_coords_and_get_time_x_y_data(
    dictionary,
    set_names,
    data_type,
    bodyparts_list_to_calculate_centroid,
    threshold_likelihood,
    threshold_distance,
    column_name_list,
):

    for set_name in set_names:

        if bodyparts_list_to_calculate_centroid is not None or data_type == "multi_bodyparts":

            data = replace_low_likelihood_coords(
                dictionary[set_name]["skeleton_data_formatted"],
                dictionary[set_name]["file_names"],
                threshold_likelihood,
            )  # NOW NO CORRECTION IS PERFORMED HERE
            dictionary[set_name]["skeleton_data_corrected"] = replace_fast_moving_coords(
                data,
                dictionary[set_name]["file_names"],
                threshold_distance,
                column_name_list=None,
            )  # NOW NO CORRECTION IS PERFORMED HERE

            dictionary[set_name]["time_x_y_data"] = get_time_x_y_data(
                dictionary[set_name]["skeleton_data_corrected"],
                data_type,
                bodyparts_list_to_calculate_centroid,
            )

        else:
            userset_x_y_corrected = replace_fast_moving_coords(
                dictionary[set_name]["userset_data_formatted"],
                dictionary[set_name]["file_names"],
                threshold_distance,
                column_name_list,
            )  # NOW NO CORRECTION IS PERFORMED HERE

            # Just reshape
            dictionary[set_name]["time_x_y_data"] = get_time_x_y_data(
                userset_x_y_corrected,
                data_type,
                bodyparts_list_to_calculate_centroid,
            )

    return dictionary


def find_max_timestamps(
    data_dict,
):

    """
    Find the longest number_of_timestamps from all arrays in time_x_y_data.
    """

    max_timestamps = 0
    for key in data_dict:
        for array in data_dict[key]["time_x_y_data"]:
            # number_of_timestamps is on the second axis
            max_timestamps = max(max_timestamps, array.shape[1])

    print(f"Maximum number_of_timestamps: {max_timestamps}")
    
    return max_timestamps


def pad_array_to_length(
    array, 
    target_length, 
    timestamp_axis, 
    padding_value=np.nan,
):

    """
    Function to pad the array with NaNs up to the specified length.
    """

    shape = list(array.shape)
    shape[timestamp_axis] = target_length  # Target the axis for number_of_timestamps
    padded_array = np.full(shape, padding_value)
    original_length = array.shape[timestamp_axis]  # Current number_of_timestamps

    if timestamp_axis == 1: 
        padded_array[:, :original_length, :] = array # Copy the original data 
    else:
        padded_array[:, :, :original_length, :] = array # Copy the original data 

    return padded_array


def pad_and_concat_data(
    data_list, 
    max_timestamps, 
    timestamp_axis,
):

    """
    Pad all data up to the maximum number_of_timestamps with NaNs
    and store the concatenated result in new keys.
    """

    # Retrieve skeleton_data, userset_data, or time_x_y_data
    padded = [pad_array_to_length(arr, max_timestamps, timestamp_axis) for arr in data_list]

    # Concatenate the NaN-padded arrays
    padded = np.array(padded)

    return padded


def convert_from_list_to_array_with_nan_padding(
    dictionary,
    set_names,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
):

    if has_both_of_skeleton_and_userset:
        data_category_names = ["onehot_labels", "skeleton_data_corrected", "userset_data_formatted", "time_x_y_data"]
    if has_only_skeleton:
        data_category_names = ["onehot_labels", "skeleton_data_corrected", "time_x_y_data"]
    if has_only_userset:
        data_category_names = ["onehot_labels", "userset_data_formatted", "time_x_y_data"]

    max_timestamps = find_max_timestamps(dictionary)

    for set_name in set_names:
        for data_category in data_category_names:

            if data_category == "onehot_labels":
                dictionary[set_name]["onehot_labels_array"] = np.array(dictionary[set_name]["onehot_labels"]).astype(np.float32)

            if data_category == "skeleton_data_corrected":
                dictionary[set_name]["skeleton_nan_padded_array"] = pad_and_concat_data(dictionary[set_name]["skeleton_data_corrected"], max_timestamps, timestamp_axis=2)

            if data_category == "userset_data_formatted":
                dictionary[set_name]["userset_nan_padded_array"] = pad_and_concat_data(dictionary[set_name]["userset_data_formatted"], max_timestamps, timestamp_axis=1)

            if data_category == "time_x_y_data":
                dictionary[set_name]["time_x_y_nan_padded_array"] = pad_and_concat_data(dictionary[set_name]["time_x_y_data"], max_timestamps, timestamp_axis=1)

    return dictionary


def calculate_speed(
    array
):
    delta_x         = np.diff(array[:, :, 1], axis=1)    # (N (number_of_animals or number_of_bodyparts), (number_of_timestamps-1))
    delta_y         = np.diff(array[:, :, 2], axis=1)    # (N, (number_of_timestamps-1))
    delta_timestamp = np.diff(array[:, :, 0], axis=1)    # (N, (number_of_timestamps-1))
    delta_distance  = np.sqrt(delta_x**2 + delta_y**2)  # (N, (number_of_timestamps-1))

    speed = delta_distance / delta_timestamp            # (N, (number_of_timestamps-1))

    # Insert a zero at the first timestamp for the missing an element due to np.diff
    speed = np.insert(speed, 0, 0, axis=1)

    return speed


def calculate_acceleration(
    array
):
    
    delta_x         = np.diff(array[:, :, 1], axis=1)    # (N, (number_of_timestamps-1))
    delta_y         = np.diff(array[:, :, 2], axis=1)    # (N, (number_of_timestamps-1))
    delta_timestamp = np.diff(array[:, :, 0], axis=1)    # (N, (number_of_timestamps-1))
    delta_distance  = np.sqrt(delta_x**2 + delta_y**2)  # (N, (number_of_timestamps-1))

    speed = delta_distance / delta_timestamp            # (N, (number_of_timestamps-1))

    # Calculate the difference in speed for acceleration
    delta_speed = np.diff(speed, axis=1)                # (N, (number_of_timestamps-2))

    # Adjust the time differences for acceleration calculation (need to skip the first time delta as we"re working with speed deltas now)
    delta_time_for_acceleration = delta_timestamp[:, 1:]   # (N, (number_of_timestamps-2))

    # Calculate acceleration as the change in speed over time
    acceleration = delta_speed / delta_time_for_acceleration  # (N, (number_of_timestamps-2))
    
    # Insert zeros at the first two timestamps for the missing two elements due to np.diff
    acceleration = np.insert(acceleration, 0, [[0], [0]], axis=1)

    return acceleration


def calculate_azimuth_and_turning_angle(
    array
):
    """
    Calculate azimuth and turning angle.

    ##### Bearing #####
    Calculate the azimuth (in radians) from point (x1, y1) to point (x2, y2), using North as the reference, in the range (0, 2π).
    The azimuth is the angle measured in a clockwise direction from the north line.

    The value of azimuth:
        North: The azimuth is 0.
        Northeast: The azimuth is π/4​ or approximately 0.79 radians.
        East: The azimuth is π/2​ or approximately 1.57 radians.
        Southeast: The azimuth is 3π/4​ or approximately 2.36 radians.
        South: The azimuth is π or approximately 3.14 radians.
        Southwest: The azimuth is 7π/4​ or approximately 3.93 radians.
        West: The azimuth is 3π/2​ or approximately 4.71 radians.
        Northwest: The azimuth is 5π/4​ or approximately 5.50 radians.
        Same location: The azimuth value from the previous timestamp.
    """

    ##### azimuth #####
    delta_x = np.diff(array[:, :, 1], axis=1)  # (N, (number_of_timestamps-1))
    delta_y = np.diff(array[:, :, 2], axis=1)  # (N, (number_of_timestamps-1))

    # Calculate the angle in radians from the north, adjusting from the positive x-axis
    azimuth = np.arctan2(delta_y, delta_x)  # (N, (number_of_timestamps-1))
    azimuth = np.pi / 2 - azimuth  # Adjust for north as reference
    azimuth = np.mod(azimuth, 2 * np.pi)  # standardize the azimuth to the range (0, 2π)

    # Insert a zero at the first timestamp for the missing an element due to np.diff
    # Use the azimuth value from the previous timestamp when two points are at the same location.
    azimuth = np.insert(azimuth, 0, 0, axis=1)
    for animal in range(azimuth.shape[0]):
        for i in range(1, azimuth.shape[1]):
            if delta_x[animal, i-1] == 0 and delta_y[animal, i-1] == 0:
                azimuth[animal, i] = azimuth[animal, i-1]

    ##### turning angle #####
    diff = np.diff(azimuth, axis=1)

    # If the difference in angles is greater than π radians, subtract 2π radians (360 degrees) for correction.
    # For example, a change from 359 degrees to 1 degree is actually a small change.
    turning_angle = np.where(diff > np.pi, diff - 2*np.pi, diff)

    # If the difference in angles is less than -π radians, add 2π radians (360 degrees) for correction.
    # For example, a change from 1 degree to 359 degrees is actually a small change.
    turning_angle = np.where(diff < -np.pi, diff + 2*np.pi, turning_angle)

    # Insert a zero at the first timestamp for the missing element due to np.diff
    turning_angle = np.insert(turning_angle, 0, 0, axis=1)

    return azimuth, turning_angle


def calculate_head_direction(
    skeleton_array,  # (number_of_animals, number_of_bodyparts, number_of_timestamps, 3(time, x, y))
    head_side_bodypart, 
    tail_side_bodypart,
):
    delta_x  = skeleton_array[:, tail_side_bodypart, :, 1] - skeleton_array[:, head_side_bodypart, :, 1]  # (N, number_of_timestamps)
    delta_y  = skeleton_array[:, tail_side_bodypart, :, 2] - skeleton_array[:, head_side_bodypart, :, 2]  # (N, number_of_timestamps)

    azimuth = np.arctan2(delta_y, delta_x)
    azimuth = np.pi / 2 - azimuth
    azimuth = np.mod(azimuth, 2 * np.pi)

    for animal in range(azimuth.shape[0]):
        for i in range(1, azimuth.shape[1]):
            if delta_x[animal, i-1] == 0 and delta_y[animal, i-1] == 0:
                azimuth[animal, i] = azimuth[animal, i-1]

    return azimuth


def calculate_distance_and_angle_from_initial_location(
    array,
):

    initial_x = array[:, 0, 1]
    initial_y = array[:, 0, 2]

    delta_x  = array[:, :, 1] - initial_x[:, np.newaxis]  # (N, number_of_timestamps)
    delta_y  = array[:, :, 2] - initial_y[:, np.newaxis]  # (N, number_of_timestamps)

    distance_from_initial_location = np.sqrt(delta_x**2 + delta_y**2)  # (N, number_of_timestamps)

    # Calculate the angle in radians from the north, adjusting from the positive x-axis
    angle = np.arctan2(delta_y, delta_x)  # (N, number_of_timestamps)
    angle = np.pi / 2 - angle  # Adjust for north as reference

    # standardize the angle to the range (0, 2π)
    angle_from_initial_location = np.mod(angle, 2 * np.pi)

    # Fix the value of the first timestep
    angle_from_initial_location[:, 0] = 0

    return distance_from_initial_location, angle_from_initial_location


def calculate_distance_and_angle_from_userdefined_location(
    array,
    userdefined_location_x,
    userdefined_location_y,
):
    """
    Calculate the distance and the angle from a specified point in the field.
    """

    delta_x  = array[:, :, 1] - userdefined_location_x  # (N, number_of_timestamps)
    delta_y  = array[:, :, 2] - userdefined_location_y  # (N, number_of_timestamps)

    distance_from_userdefined_location  = np.sqrt(delta_x**2 + delta_y**2)  # (N, number_of_timestamps)

    # Calculate the angle in radians from the north, adjusting from the positive x-axis
    angle = np.arctan2(delta_y, delta_x)  # (N, number_of_timestamps)
    angle = np.pi / 2 - angle  # Adjust for north as reference

    # standardize the angle to the range (0, 2π)
    angle_from_userdefined_location = np.mod(angle, 2 * np.pi)

    return distance_from_userdefined_location, angle_from_userdefined_location


def calculate_cumulative_travel_distance(
    array
):

    distance = np.sqrt((np.diff(array[:, :, 1], axis=1))**2 + (np.diff(array[:, :, 2], axis=1))**2)  # (N, number_of_timestamps)

    cumulative_travel_distance = np.cumsum(distance, axis=1)

    # Insert a zero at the first timestamp for the missing an element due to np.diff
    cumulative_travel_distance = np.insert(cumulative_travel_distance, 0, 0, axis=1)

    return cumulative_travel_distance


def calculate_cumulative_change_in_turning_angle(
    turning_angle
):

    absolute_value_of_diff = np.abs(np.diff(turning_angle, axis=1))

    # Find indices where turning_angle (excluding the first element) equals 0
    zero_indices = np.where(turning_angle[:, 1:] == 0)

    # Set the corresponding absolute_value_of_diff values to 0 for indices found
    for i, j in zip(*zero_indices):
        absolute_value_of_diff[i, j] = 0

    cumulative_change_in_turning_angle = np.cumsum(absolute_value_of_diff, axis=1)

    # Insert a zero at the first timestamp for the missing an element due to np.diff
    cumulative_change_in_turning_angle = np.insert(cumulative_change_in_turning_angle, 0, 0, axis=1)

    return cumulative_change_in_turning_angle


def calculate_features(
    dictionary,
    set_names,
    column_name_list,
    has_both_of_skeleton_and_userset,
    has_only_skeleton,
    has_only_userset,
    userdefined_location_x,
    userdefined_location_y,
    head_side_bodypart,
    tail_side_bodypart,
):
    """
    Return:
    features: array
        array shape:
            single: (number_of_files, 1, number_of_timestamps, Features)
            multi_animals: (number_of_files, number_of_animals, number_of_timestamps, Features)
            multi_bodyparts: (number_of_files, number_of_bodyparts, number_of_timestamps, Features)
    feature_name_list: length Features
    """

    ##### Make a name list of data #####
    feature_name_list = [
        "time",
        "x",
        "y",
        "Speed",
        "Acceleration",
        "Azimuth",
        "Turning angle",
        "Distance from initial location",
        "Angle from initial location",
        "Distance from userdefined location",
        "Angle from userdefined location",
        "Cumulative travel distance",
        "Cumulative change in turning angle",
    ]

    if not(head_side_bodypart == -1 and tail_side_bodypart == -1):
        feature_name_list.append("Head direction")

    if has_both_of_skeleton_and_userset or has_only_userset:
        excluded_columns = {"time", "x", "y"}  # Do not directory use time, x, y columns in userset data
        selected_indices = [i for i, col in enumerate(column_name_list) if col not in excluded_columns]
        if selected_indices:
            feature_name_list.extend([column_name_list[i] for i in selected_indices])

    number_of_features = len(feature_name_list)

    ##### Calculate features #####
    for set_name in set_names:
        data = dictionary[set_name]["time_x_y_nan_padded_array"]
        number_of_files, number_of_nodes, number_of_timestamps, _ = data.shape
        features = np.full((number_of_files, number_of_nodes, number_of_timestamps, number_of_features), np.nan, dtype=np.float32)

        for i, array in enumerate(data):
            # Create a mask for non-NaN parts (using only one sample to determine valid length)
            valid_mask = ~np.isnan(array[0, :, 0])  # Check non-NaN parts in the first sample

            # Get the index where the first False (NaN) appears
            max_valid_length = np.argmax(~valid_mask) if np.any(~valid_mask) else len(valid_mask)

            # Slice the array up to the maximum valid length
            array = array[:, :max_valid_length, :]
            
            time = array[:, :, 0]
            x = array[:, :, 1]
            y = array[:, :, 2]
            speed = calculate_speed(array)
            acceleration = calculate_acceleration(array)
            azimuth, turning_angle = calculate_azimuth_and_turning_angle(array)
            distance_from_initial_location, angle_from_initial_location = calculate_distance_and_angle_from_initial_location(array)
            distance_from_userdefined_location, angle_from_userdefined_location = calculate_distance_and_angle_from_userdefined_location(array, userdefined_location_x, userdefined_location_y)
            cumulative_travel_distance = calculate_cumulative_travel_distance(array)
            cumulative_change_in_turning_angle = calculate_cumulative_change_in_turning_angle(turning_angle)

            feature_data = np.stack((
                time,
                x,
                y,
                speed,
                acceleration,
                azimuth,
                turning_angle,
                distance_from_initial_location,
                angle_from_initial_location,
                distance_from_userdefined_location,
                angle_from_userdefined_location,
                cumulative_travel_distance,
                cumulative_change_in_turning_angle,
            ), axis=2)

            if not(head_side_bodypart == -1 and tail_side_bodypart == -1):
                skltn_array = dictionary[set_name]["skeleton_nan_padded_array"][i]
                skltn_array = skltn_array[:, :max_valid_length, :]
                head_direction = calculate_head_direction(skltn_array, head_side_bodypart, tail_side_bodypart)
                feature_data = np.concatenate((feature_data, head_direction[:, :, np.newaxis]), axis=2)

            if has_both_of_skeleton_and_userset or has_only_userset:
                if selected_indices:
                    # Avoid transposition due to advanced indexing
                    userset_f = np.take(
                        dictionary[set_name]["userset_nan_padded_array"][i, :, :max_valid_length, :], 
                        selected_indices, 
                        axis=2
                    )
                    feature_data = np.concatenate((feature_data, userset_f), axis=2)

            features[i, :, :max_valid_length, :] = feature_data.astype(np.float32)

        dictionary[set_name]["features"] = features

    return dictionary, feature_name_list


def calculate_pose_feature_convolution(
    dictionary,
    set_names,
    bodyparts_list_for_pose,
):
    """
    return: array
        array shape:
            single: (number_of_files, 1, number_of_timestamps, number_of_channels_for_skeleton)
            multi_animals: (number_of_files, number_of_animals, number_of_timestamps, number_of_channels_for_skeleton)
            multi_bodyparts: (number_of_files, number_of_bodyparts, number_of_timestamps, 1)
    """

    for set_name in set_names:

        if bodyparts_list_for_pose is None:
            number_of_files, number_of_animals, number_of_timestamps, _ = dictionary[set_name]["time_x_y_nan_padded_array"].shape
            dictionary[set_name]["pose_feature"] = np.full((number_of_files, number_of_animals, number_of_timestamps, 1), None, dtype=object)

        elif bodyparts_list_for_pose == "Default":
            skeleton_data = dictionary[set_name]["skeleton_nan_padded_array"]
            number_of_files, number_of_animals, number_of_bodyparts, number_of_timestamps, _ = skeleton_data.shape
            pose_feature = np.full((number_of_files, number_of_animals, number_of_timestamps, int(number_of_bodyparts*(number_of_bodyparts-1)/2)), np.nan, dtype=np.float32)

            for idx, array in enumerate(skeleton_data):
                # Create a mask for non-NaN parts (using only one sample to determine valid length)
                valid_mask = ~np.isnan(array[0, 0, :, 0])  # Check non-NaN parts in the first sample

                # Get the index where the first False (NaN) appears
                max_valid_length = np.argmax(~valid_mask) if np.any(~valid_mask) else len(valid_mask)

                # Slice the array up to the maximum valid length
                array = array[:, :, :max_valid_length, :]

                number_of_animals, number_of_bodyparts, number_of_timestamps, _ = array.shape  # (number_of_animals, number_of_bodyparts, number_of_timestamps, 3)
                centroids = np.mean(array[:, :, :, 1:3], axis=1, keepdims=True)  # (N, 1, number_of_timestamps, 2)
                vectors = centroids - array[:, :, :, 1:3]  # (N, number_of_bodyparts, number_of_timestamps, 2)
                combs = list(combinations(range(number_of_bodyparts), 2))
                angles = np.zeros((number_of_animals, number_of_timestamps, len(combs)))

                for i, (a, b) in enumerate(combs):
                    vec_a = vectors[:, a, :, :]  # (N, number_of_timestamps, 2)
                    vec_b = vectors[:, b, :, :]  # (N, number_of_timestamps, 2)

                    dot_product = np.sum(vec_a * vec_b, axis=2)  # (N, number_of_timestamps)
                    norm_a = np.linalg.norm(vec_a, axis=2)  # (N, number_of_timestamps)
                    norm_b = np.linalg.norm(vec_b, axis=2)  # (N, number_of_timestamps)
                    cosine_angle = dot_product / (norm_a * norm_b)

                    # angles (N, number_of_timestamps, len(combs)
                    angles[:, :, i] = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # clip for safe

                pose_feature[idx, :, :max_valid_length, :] = angles.astype(np.float32)
            
            dictionary[set_name]["pose_feature"] = pose_feature
        
        else:
            skeleton_data = dictionary[set_name]["skeleton_nan_padded_array"]
            number_of_files, number_of_animals, number_of_bodyparts, number_of_timestamps, _ = skeleton_data.shape
            pose_feature = np.full((number_of_files, number_of_animals, number_of_timestamps, len(bodyparts_list_for_pose)), np.nan, dtype=np.float32)

            for idx, array in enumerate(skeleton_data):
                # Create a mask for non-NaN parts (using only one sample to determine valid length)
                valid_mask = ~np.isnan(array[0, 0, :, 0])  # Check non-NaN parts in the first sample

                # Get the index where the first False (NaN) appears
                max_valid_length = np.argmax(~valid_mask) if np.any(~valid_mask) else len(valid_mask)

                # Slice the array up to the maximum valid length
                array = array[:, :, :max_valid_length, :]

                number_of_animals, _, number_of_timestamps, _ = array.shape
                angles = np.zeros((number_of_animals, number_of_timestamps, len(bodyparts_list_for_pose)))

                for i, comb in enumerate(bodyparts_list_for_pose):
                    center_index = comb[1]
                    index1 = comb[0]
                    index2 = comb[2]

                    vec_a = array[:, center_index, :, 1:3] - array[:, index1, :, 1:3]
                    vec_b = array[:, center_index, :, 1:3] - array[:, index2, :, 1:3]

                    dot_product = np.sum(vec_a * vec_b, axis=2)
                    norm1 = np.linalg.norm(vec_a, axis=2)
                    norm2 = np.linalg.norm(vec_b, axis=2)
                    cosine_angle = dot_product / (norm1 * norm2)
                    angles[:, :, i] = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

                pose_feature[idx, :, :max_valid_length, :] = angles.astype(np.float32)
            
            dictionary[set_name]["pose_feature"] = pose_feature

    return dictionary


def calculate_distance_between_nodes(
    edge_idx, 
    time_x_y_array,
):
    """
    Return: array
    array shape:
        (E(number of edges. combination of nodes), number_of_timestamps, 1)
    """

    _, number_of_edges = edge_idx.shape  # (2, number_of_edges)
    _, number_of_timestamps, _ = time_x_y_array.shape  # (N, number_of_timestamps, 3(time, x, y))

    distance = np.zeros((number_of_edges, number_of_timestamps, 1))

    for i, edge in enumerate(edge_idx.transpose()):

        node_0_x = time_x_y_array[edge[0], :, 1]
        node_0_y = time_x_y_array[edge[0], :, 2]
        node_1_x = time_x_y_array[edge[1], :, 1]
        node_1_y = time_x_y_array[edge[1], :, 2]

        delta_x  = node_0_x - node_1_x
        delta_y  = node_0_y - node_1_y

        dist = np.sqrt(delta_x**2 + delta_y**2)

        distance[i, :, 0] = dist

    return distance


def calculate_angle_difference_between_nodes(
    edge_idx, 
    features_array,
    angle_index,
):
    """Angular difference between the two nodes of each edge.

    The difference is wrapped into [-pi, pi], so that 1 degree against 359
    degrees reads as the small difference it is. Every statistic taken over
    these values downstream (mean, variance, quantile aggregates) therefore
    works on wrapped differences and needs no further circular treatment.

    Return: array
    array shape:
        (E(number of edges. combination of nodes), number_of_timestamps, 1)
    """

    _, number_of_edges = edge_idx.shape  # (2, number_of_edges)
    _, number_of_timestamps, _ = features_array.shape  # (N, number_of_timestamps, Features)

    array = features_array[:, :, angle_index]
    
    angle_difference_between_nodes = np.zeros((number_of_edges, number_of_timestamps, 1))

    for i, edge in enumerate(edge_idx.transpose()):
        node_0 = array[edge[0], :]
        node_1 = array[edge[1], :]

        diff = node_0 - node_1

        # If the difference in angles is greater than π radians, subtract 2π radians (360 degrees) for correction.
        # For example, a change from 359 degrees to 1 degree is actually a small change.
        angle_diff = np.where(diff > np.pi, diff - 2*np.pi, diff)

        # If the difference in angles is less than -π radians, add 2π radians (360 degrees) for correction.
        # For example, a change from 1 degree to 359 degrees is actually a small change.
        angle_diff = np.where(diff < -np.pi, diff + 2*np.pi, angle_diff)

        angle_difference_between_nodes[i, :, 0] = angle_diff
                
    return angle_difference_between_nodes


def calculate_bodypart_to_bodypart_distance_between_nodes(
    edge_idx, 
    skeleton_data_array,
    bodyparts_pair_list,
):
    
    """
    Return: array
    array shape:
        (E(number of edges. combination of nodes), number_of_timestamps, 1)
    """

    _, number_of_edges = edge_idx.shape  # (2, number_of_edges)
    _, _, number_of_timestamps, _ = skeleton_data_array.shape  # multi_animals: (number_of_animals, number_of_bodyparts, number_of_timestamps, 3(time, x, y))

    distance = np.zeros((number_of_edges, number_of_timestamps, len(bodyparts_pair_list)))

    for i, edge in enumerate(edge_idx.transpose()):

        for j, bodyparts_pair in enumerate(bodyparts_pair_list):  # Expect bi-directional graph

            node_0_bodypart_0_x = skeleton_data_array[edge[0], bodyparts_pair[0], :, 1]
            node_0_bodypart_0_y = skeleton_data_array[edge[0], bodyparts_pair[0], :, 2]
            node_1_bodypart_1_x = skeleton_data_array[edge[1], bodyparts_pair[1], :, 1]
            node_1_bodypart_1_y = skeleton_data_array[edge[1], bodyparts_pair[1], :, 2]

            delta_x  = node_0_bodypart_0_x - node_1_bodypart_1_x
            delta_y  = node_0_bodypart_0_y - node_1_bodypart_1_y

            dist = np.sqrt(delta_x**2 + delta_y**2)

            distance[i, :, j] = dist
                
    return distance


def calculate_edge_data(
    dictionary,
    set_names,
    edge_index_one_datapoint,
    feature_name_list,
    bodyparts_pair_list,
):

    # Make a name list of data
    edge_data_name_list = [
        "Distance between nodes", 
        "Azimuth difference between nodes", 
    ]

    try:
        head_direction_index = feature_name_list.index("Head direction")
    except ValueError:
        head_direction_index = None

    if head_direction_index is not None:
        edge_data_name_list.append("Head direction difference between nodes")

    if bodyparts_pair_list is not None:
        for pair in bodyparts_pair_list:
            bodypart_1, bodypart_2 = pair
            dynamic_name = f"Bodypart{bodypart_1+1}-to-bodypart{bodypart_2+1} distance between nodes"
            edge_data_name_list.append(dynamic_name)

    number_of_features = len(edge_data_name_list)

    # Calculate features
    for set_name in set_names:
        time_x_y_data = dictionary[set_name]["time_x_y_nan_padded_array"]
        skeleton_data = dictionary[set_name]["skeleton_nan_padded_array"]
        features = dictionary[set_name]["features"]

        number_of_files, number_of_nodes, number_of_timestamps, _ = features.shape

        dictionary[set_name]["edge_index"] = np.array([edge_index_one_datapoint.copy() for _ in range(number_of_files)])  # (number_of_files, 2, E(number of edges. combination of nodes))
        edge_index = dictionary[set_name]["edge_index"]

        _, _, number_of_edges = edge_index.shape
        all_edge_data = np.full((number_of_files, number_of_edges, number_of_timestamps, number_of_features), np.nan, dtype=np.float32)

        for i in range(number_of_files):

            # Create a mask for non-NaN parts (using only one sample to determine valid length)
            valid_mask = ~np.isnan(time_x_y_data[i, 0, :, 0])  # Check non-NaN parts in the first sample

            # Get the index where the first False (NaN) appears
            max_valid_length = np.argmax(~valid_mask) if np.any(~valid_mask) else len(valid_mask)

            distance_between_nodes = calculate_distance_between_nodes(edge_index[i], time_x_y_data[i, :, :max_valid_length, :])

            azimuth_index = feature_name_list.index("Azimuth")
            azimuth_difference_between_nodes = calculate_angle_difference_between_nodes(edge_index[i], features[i, :, :max_valid_length, :], azimuth_index)
        
            edge_data = np.concatenate((
                distance_between_nodes,
                azimuth_difference_between_nodes,
            ), axis=2)
            
            if head_direction_index is not None:
                head_direction_difference_between_nodes = calculate_angle_difference_between_nodes(edge_index[i], features[i, :, :max_valid_length, :], head_direction_index)
                edge_data = np.concatenate((edge_data, head_direction_difference_between_nodes), axis=2)

            if bodyparts_pair_list is not None:
                bodypart_to_bodypart_distance_between_nodes = calculate_bodypart_to_bodypart_distance_between_nodes(edge_index[i], skeleton_data[i, :, :, :max_valid_length, :], bodyparts_pair_list)
                edge_data = np.concatenate((edge_data, bodypart_to_bodypart_distance_between_nodes), axis=2)

            all_edge_data[i, :, :max_valid_length, :] = edge_data.astype(np.float32)

        dictionary[set_name]["edge_data"] = all_edge_data

    return dictionary, edge_data_name_list


def select_features_for_machine_learning(
    dictionary, 
    set_names,
    feature_name_list,
    feature_name_list_for_machine_learning,
    edge_data_name_list,
    edge_data_name_list_for_machine_learning,
):

    # Get feature indices
    feature_indices_for_machine_learning = [feature_name_list.index(item) for item in feature_name_list_for_machine_learning]

    if edge_data_name_list is None:
        edge_data_indices_for_machine_learning = None
    else:
        edge_data_indices_for_machine_learning = [edge_data_name_list.index(item) for item in edge_data_name_list_for_machine_learning]

    # Get selected features
    for set_name in set_names:
        dictionary[set_name]["features_for_machine_learning"] = dictionary[set_name]["features"][:, :, :, feature_indices_for_machine_learning]
        if edge_data_name_list is None:
            number_of_files, *_ = dictionary[set_name]["features"].shape
            dictionary[set_name]["edge_data_for_machine_learning"] = np.full((number_of_files), None, dtype=object)
        else:
            dictionary[set_name]["edge_data_for_machine_learning"] = dictionary[set_name]["edge_data"][:, :, :, edge_data_indices_for_machine_learning]

    return dictionary


def standardize_and_replace_nan_to_zero(
    dictionary,
    set_names,
):
    """
    Standardize trajectoryset, ignoring NaN. Then replace NaN to zero.

    Use the same scaling method for all features.
    https://stackoverflow.com/questions/61530077/is-it-right-to-use-different-feature-scaling-techniques-to-different-features
    """

    # Calculate mean and std of each features from the train set
    if np.all(dictionary["train"]["pose_feature"] == None):
        train_mean_pose_feature = None
        train_std_pose_feature = None
    else:
        train_mean_pose_feature = np.nanmean(dictionary["train"]["pose_feature"])
        train_std_pose_feature = np.nanstd(dictionary["train"]["pose_feature"])

    train_mean_features = np.nanmean(dictionary["train"]["features_for_machine_learning"], axis=(0, 1, 2), keepdims=True)
    train_std_features = np.nanstd(dictionary["train"]["features_for_machine_learning"], axis=(0, 1, 2), keepdims=True)

    if np.all(dictionary["train"]["edge_data_for_machine_learning"] == None):
        train_mean_edge = None
        train_std_edge = None
    else:
        train_mean_edge = np.nanmean(dictionary["train"]["edge_data_for_machine_learning"], axis=(0, 1, 2), keepdims=True)
        train_std_edge = np.nanstd(dictionary["train"]["edge_data_for_machine_learning"], axis=(0, 1, 2), keepdims=True)

    # Standardize
    for set_name in set_names:
        if train_mean_pose_feature is None:
            dictionary[set_name]["pose_feature_standardized"] = dictionary[set_name]["pose_feature"]
        else:
            normalized_array = (dictionary[set_name]["pose_feature"] - train_mean_pose_feature) / train_std_pose_feature
            dictionary[set_name]["pose_feature_standardized"] = normalized_array.astype(np.float32)

        normalized_array = (dictionary[set_name]["features_for_machine_learning"] - train_mean_features) / train_std_features
        dictionary[set_name]["features_standardized"] = normalized_array.astype(np.float32)

        if train_mean_edge is None:
            dictionary[set_name]["edge_data_standardized"] = dictionary[set_name]["edge_data_for_machine_learning"]
        else:
            normalized_array = (dictionary[set_name]["edge_data_for_machine_learning"] - train_mean_edge) / train_std_edge
            dictionary[set_name]["edge_data_standardized"] = normalized_array.astype(np.float32)

    # Save mean and std for train set
    keys = ["pose_feature_standardized", "features_standardized", "edge_data_standardized"]
    mean_and_std_of_train_dictionary = {
        key: {"mean": None, "std": None} for key in keys
    }

    mean_and_std_of_train_dictionary["pose_feature_standardized"]["mean"] = train_mean_pose_feature
    mean_and_std_of_train_dictionary["pose_feature_standardized"]["std"] = train_std_pose_feature
    mean_and_std_of_train_dictionary["features_standardized"]["mean"] = train_mean_features
    mean_and_std_of_train_dictionary["features_standardized"]["std"] = train_std_features
    mean_and_std_of_train_dictionary["edge_data_standardized"]["mean"] = train_mean_edge
    mean_and_std_of_train_dictionary["edge_data_standardized"]["std"] = train_std_edge

    # Replace nan to zero
    zero_count = 0
    for set_name in ["train", "val", "test"]:
        for key in keys:
            zero_count += np.count_nonzero(dictionary[set_name][key] == 0)  # in NumPy, 0 == 0.0
            dictionary[set_name][key] = np.nan_to_num(dictionary[set_name][key], nan=0.0)
    print("number of zero count over the train, val, and test sets:", zero_count)

    return dictionary, mean_and_std_of_train_dictionary


def get_weighted_adjacency(num_node, edge_idx, edge_w, fill):
    adjacency = np.full((num_node, num_node), fill, dtype=np.float32)
    for idx in range(edge_idx.shape[1]):
        i = edge_idx[0, idx]
        j = edge_idx[1, idx]
        w = edge_w[idx]
        adjacency[i, j] = w
    return adjacency


def calculate_weighted_adjacency_matrix(
    dictionary,
    set_names,
):

    for set_name in set_names:
        x = dictionary[set_name]["features_standardized"]
        edge_index = dictionary[set_name]["edge_index"]
        edge_data = dictionary[set_name]["edge_data_standardized"]

        if np.all(edge_data == None):
            dictionary[set_name]["edge_attributes"] = edge_data

        else:
            _, number_of_nodes, _, _ = x.shape
            number_of_files, _, number_of_timestamps, number_of_dimensions = edge_data.shape
    
            weighted_adjacency_matrix = np.zeros((number_of_files, number_of_nodes, number_of_nodes, number_of_timestamps, number_of_dimensions), dtype=np.float32)
            for f in range(number_of_files):
                edge_idx = edge_index[f]
                for c in range(number_of_dimensions):
                    for t in range(number_of_timestamps):
                        adjacency = get_weighted_adjacency(number_of_nodes, edge_idx, edge_data[f, :, t, c].astype(np.float32), 0.0)
                        weighted_adjacency_matrix[f, :, :, t, c] = adjacency

                print(set_name, f, "/", number_of_files, "finished.")

            dictionary[set_name]["edge_attributes"] = weighted_adjacency_matrix.astype(np.float32)

    return dictionary


def valid_convolve(xx, size):
    """
    This function provides a moving average calculation with adjustments for edge cases based on np.convolve.
    https://zenn.dev/bluepost/articles/1b7b580ab54e95
    """

    # Create a window for the moving average calculation
    b = np.ones(size) / size
    n_conv = math.ceil(size / 2)
    xx_mean = np.empty_like(xx).astype(np.float32)

    # Apply convolution for each entity and feature
    for n in range(xx.shape[0]):  # Loop over the number of entities
        for f in range(xx.shape[2]):  # Loop over the number of features
            temp_mean = np.convolve(xx[n, :, f], b, mode="same").astype(np.float32)
            
            # Adjust for edge cases
            temp_mean[0] *= size / n_conv
            for i in range(1, n_conv):
                temp_mean[i] *= size / (i + n_conv)
                temp_mean[-i] *= size / (i + n_conv - (size % 2))
            
            xx_mean[n, :, f] = temp_mean.astype(np.float32)

    return xx_mean


# Features whose values are angles in radians. Their moving statistics are
# computed on the circle: a linear mean of 0.05 and 6.23 rad would land near
# pi, on the opposite side of the circle from both inputs.
#
# "Cumulative change in turning angle" is deliberately absent: it accumulates
# absolute differences and grows past 2*pi, so it is an ordinary quantity.
ANGULAR_FEATURE_NAMES = (
    "Azimuth",
    "Turning angle",
    "Angle from initial location",
    "Angle from userdefined location",
    "Head direction",
)

MOVING_AVERAGE_SUFFIX = " (moving average)"
MOVING_VARIANCE_SUFFIX = " (moving variance)"


def is_angular_feature(
    feature_name,
):
    """Whether a feature holds angles in radians.

    A moving average of an angular feature is still an angle, but its circular
    variance is a ratio in [0, 1] and so is not angular.
    """
    if feature_name.endswith(MOVING_VARIANCE_SUFFIX):
        return False
    if feature_name.endswith(MOVING_AVERAGE_SUFFIX):
        feature_name = feature_name[: -len(MOVING_AVERAGE_SUFFIX)]
    return feature_name in ANGULAR_FEATURE_NAMES


def circular_moving_average_and_variance(
    angles,
    moving_window_size,
):
    """Moving circular mean and circular variance of angles in radians.

    The mean is atan2 of the moving averages of sin and cos, wrapped to
    [0, 2*pi). The variance is 1 - R, where R is the length of the mean
    resultant vector: 0 when the angles all agree and 1 when they cancel out.

    Args:
        angles: (number_of_nodes, number_of_timestamps, number_of_features)
    """
    sin_mean = valid_convolve(np.sin(angles), moving_window_size)
    cos_mean = valid_convolve(np.cos(angles), moving_window_size)

    circular_mean = np.mod(np.arctan2(sin_mean, cos_mean), 2 * np.pi)
    resultant_length = np.hypot(sin_mean, cos_mean)
    # Rounding can push the length slightly past 1 and make the variance negative.
    circular_variance = 1.0 - np.clip(resultant_length, 0.0, 1.0)

    return circular_mean, circular_variance


# Keys the post-analysis file needs. Anything else the pipeline leaves in the
# test dictionary is intermediate and is dropped.
POST_ANALYSIS_KEYS = (
    "file_names",
    "onehot_labels_array",
    "features_for_post_analysis",
    "edge_index",
    "edge_data",
    "skeleton_xy",
)


def build_post_analysis_dictionary(
    test_dictionary,
):
    """Trim the test dictionary down to what the post-analysis actually reads.

    The skeleton is kept as (x, y) only, since that is all the interactive
    plots use, and the coordinates and edge features are stored as float32.

    Entry i of every value belongs to file_names[i]:

        file_names                  list of length number_of_files
        onehot_labels_array         (files, classes)
        features_for_post_analysis  (files, features, nodes, timestamps)
        edge_index                  (files, 2, edges), the same graph repeated
        edge_data                   (files, edge features, edges, timestamps)
        skeleton_xy                 (files, nodes, joints, timestamps, 2)

    The caller is expected to have moved the channel axis to position 1
    beforehand, which is what the dataset notebook's transpose step does.
    """
    post_analysis = {}

    for key in ("file_names", "onehot_labels_array", "features_for_post_analysis", "edge_index"):
        if key in test_dictionary:
            post_analysis[key] = test_dictionary[key]

    edge_data = test_dictionary.get("edge_data")
    if edge_data is not None and getattr(edge_data, "dtype", None) is not None and edge_data.dtype != object:
        edge_data = edge_data.astype(np.float32)
    post_analysis["edge_data"] = edge_data

    skeleton = test_dictionary.get("skeleton_nan_padded_array")
    if skeleton is not None and getattr(skeleton, "dtype", None) is not None and skeleton.dtype != object:
        # (files, N, joints, T, (likelihood, x, y)) -> (files, N, joints, T, (x, y))
        post_analysis["skeleton_xy"] = skeleton[:, :, :, :, 1:3].astype(np.float32)

    return post_analysis


def calculate_moving_average_and_variance(
    dictionary,
    set_names,
    moving_window_size,
    feature_name_list,
):

    ##### Make a name list of data #####
    mov_avg_feature_names = [name + MOVING_AVERAGE_SUFFIX for name in feature_name_list]
    mov_var_feature_names = [name + MOVING_VARIANCE_SUFFIX for name in feature_name_list]
    feature_name_list_for_post_analysis = feature_name_list + mov_avg_feature_names + mov_var_feature_names

    # Angular features get a circular mean and a circular variance instead.
    angular_indices = [i for i, name in enumerate(feature_name_list) if is_angular_feature(name)]

    ##### Calculate features #####
    for set_name in set_names:
        data = dictionary[set_name]["features"]
        number_of_files, number_of_nodes, number_of_timestamps, number_of_features = data.shape
        mov_avg_features = np.full((number_of_files, number_of_nodes, number_of_timestamps, number_of_features), np.nan, dtype=np.float32)
        mov_var_features = np.full((number_of_files, number_of_nodes, number_of_timestamps, number_of_features), np.nan, dtype=np.float32)

        for i, array in enumerate(data):
            # Create a mask for non-NaN parts (using only one sample to determine valid length)
            valid_mask = ~np.isnan(array[0, :, 0])  # Check non-NaN parts in the first sample

            # Get the index where the first False (NaN) appears
            max_valid_length = np.argmax(~valid_mask) if np.any(~valid_mask) else len(valid_mask)

            # Slice the array up to the maximum valid length
            array = array[:, :max_valid_length, :]

            # Calculate the moving average using the adjusted convolution function
            moving_average = valid_convolve(array, moving_window_size)
            # Calculate the moving average of the squared array
            squared = np.power(array, 2)
            squared_moving_average = valid_convolve(squared, moving_window_size)

            # Calculate the variance
            moving_average_squared = np.power(moving_average, 2)
            moving_variance = squared_moving_average - moving_average_squared

            if angular_indices:
                circular_mean, circular_variance = circular_moving_average_and_variance(
                    array[:, :, angular_indices], moving_window_size
                )
                moving_average[:, :, angular_indices] = circular_mean
                moving_variance[:, :, angular_indices] = circular_variance

            mov_avg_features[i, :, :max_valid_length, :] = moving_average.astype(np.float32)
            mov_var_features[i, :, :max_valid_length, :] = moving_variance.astype(np.float32)

        dictionary[set_name]["features_for_post_analysis"] = np.concatenate((data, mov_avg_features, mov_var_features), axis=3).astype(np.float32)

    return dictionary, feature_name_list_for_post_analysis


