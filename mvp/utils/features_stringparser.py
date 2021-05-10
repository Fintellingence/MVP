""" Refined data input string parser

In this module, some functions are provided to analyze an arbitrary
string and check if it is suitable to use as input for `RefinedData`
class constructor to automatic set in cache memory useful features
that will save computational effort

This module provide the following functions to analyze strings

`get_features_iterable(input_string -> str, target -> str) -> iter`
    Return a 3-fold(zip) iterable processing the packed values
    given in `input_string`. The 3-fold parameters are
    1. ``RefinedData`` method corresponding to some feature
    2. ``tuple`` with required arguments to pass as `*args`
    3. ``dict`` with keyword-arguments to pass as `**kwargs`

`validate_features_string(input_string -> str, bar_type -> str) -> str`
    Return `input_string` corrected removing repetitions
    Raise a `ValueError` if `input_string` is ill-formatted

`extract_intervals_list(input_string -> str) -> list`
    Return list with all intervals within `input_string`
    Desirable that `input_string` be taken after passing
    in `validate_data_string` to avoid unexpected behavior

"""


def get_features_iterable(input_string, target="time:close"):
    """
    Generate iterable with 3 fields: method_name, args, kwargs
    to be used when calling `RefinedData` methods as
    `RefinedData.__getattribute__(method_name)(*args, **kwargs)

    Parameters
    ----------
    `input_string` : ``str``
        Formatted string according to the convention:
        "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."
        where KEYj must be a ``RefinedData`` method suffix starting
        with "get_", Tj the data bar interval according to `bar_type`
        and Vji represent a list of the first positional argument to
        compute the feature. In case the method requires multiple
        positional arguments Vij must be replaced by a tuple format
        to unpack as *(Vij) and must be separate by forward slash '/'
        instead of comma
    `target` : ``str``
        data target over which the feature will be computed included in kwargs

    Return
    ------
    ``iterable``
        A 3-fold iterable in the following order to unpack
        (method_name, args, kwargs) where one can call as:
        RefinedData.__getattribute__(method_name)(*args, **kwargs)

    """
    target = target.lower()
    bar_type = target.split(":")[0]
    input_string = validate_features_string(input_string.lower(), bar_type)
    attr_name_list = []
    kwargs_list = []
    args_list = []
    key_value_list_split = input_string.split(":")
    keys = key_value_list_split[::2]
    str_vals = key_value_list_split[1::2]
    for key, str_val in zip(keys, str_vals):
        attr_name = "get_" + key.split("_")[0]
        step = key.split("_")[1]
        if step != "day":
            step = int(step)
        if str_val[0] == "(":
            new_args = []
            str_tuples = str_val.split("/")
            for str_tuple in str_tuples:
                if not str_tuple:
                    continue
                single_tuple = tuple(map(int, str_tuple[1:-1].split(",")))
                new_args.append(single_tuple)
        else:
            try:
                num_vals = list(map(int, set(str_val.split(","))))
            except Exception:
                num_vals = list(map(float, set(str_val.split(","))))
            num_vals.sort()
            new_args = [(val,) for val in num_vals]
        n_vals = len(new_args)
        args_list.extend(new_args)
        kwargs_list.extend(
            n_vals * [{"step": step, "target": target, "append": True}]
        )
        attr_name_list.extend(n_vals * [attr_name])
    return zip(attr_name_list, args_list, kwargs_list)


def validate_features_string(input_string, bar_type="time"):
    """
    validate the `input_string` with set of parameters for features

    Parameters
    ----------
    `input_string` : ``str``
        string to be checked if is according to expected convention:
        "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."
        where KEYj must be a ``RefinedData`` method suffix starting
        with "get_", Tj the data bar interval according to `bar_type`
        and Vji represent a list of the first positional argument to
        compute the feature. In case the method requires multiple
        positional arguments Vij must be replaced by a tuple format
        to unpack as *(Vij) and must be separate by forward slash '/'
        instead of comma
    `bar_type` : ``str``
        Type of bar the input string refers to according to the ones
        available in `RawData` class. Valid values are attributes in
        `RawData` class with `_bars` as suffix

    Return
    ------
    ``str``
        `input_string` removing repetitions and sorting the parameters

    """
    if input_string is None:
        raise ValueError(
            "Expected string but {} given in assert data function".format(
                input_string
            )
        )
    if not input_string:
        return input_string
    key_value_list_split = input_string.split(":")
    if len(key_value_list_split) % 2 != 0:
        raise ValueError(
            "Wrong pairings divided by colons in '{}'".format(input_string)
        )
    keys = key_value_list_split[::2]
    str_vals = key_value_list_split[1::2]
    for key, str_val in zip(keys, str_vals):
        if not str_val or not key:
            raise ValueError(
                "empty fields separated by : in '{}'".format(input_string)
            )
        if len(key.split("_")) != 2:
            raise ValueError(
                "Expected feature-code to have '_' separating feature name "
                "abbreviation and interval. Check '{}'".format(input_string)
            )
        if str_val.count("(") != str_val.count(")"):
            raise ValueError(
                "Unbalanced parentheses found in '{}'".format(input_string)
            )
    __assert_intervals(input_string, bar_type)
    return __remove_repetitions(input_string.lower())


def extract_intervals_list(input_string, check_string=False, bar_type="time"):
    """
    Return list with intervals requested in formatted `input_string`

    Parameters
    ----------
    `input_string` : ``str``
        string to retrieve intervals in the convention:
        "KEY1_T1:V11,V12,...:KEY2_T2:V21,V22,...:...:KEYM_TM:VM1,..."
        where KEYj must be a ``RefinedData`` method suffix starting
        with "get_", Tj the data bar interval according to `bar_type`
        and Vji represent a list of the first positional argument to
        compute the feature. In case the method requires multiple
        positional arguments Vij must be replaced by a tuple format
        to unpack as *(Vij) and must be separate by forward slash '/'
        instead of comma
    `check_string` : ``bool``
        whether to also check if the string is formatted as required
        WARNING: if true it possibly modifies the `input_string` due
        to changes required in `validate_data_string` function
    `bar_type` : ``str``
        only required if `check_string` is True to call `validate_data_string`

    Return
    ------
    ``list``
        Return list with all intervals found in `input_string`
        `[T1, T2, ..., TM]`

    """
    if check_string:
        input_string = validate_features_string(input_string, bar_type)
    intervals = []
    if not input_string:
        return intervals
    key_interval_codes = input_string.split(":")[::2]
    string_intervals = [key.split("_")[1] for key in key_interval_codes]
    for string_interval in string_intervals:
        try:
            time_interval = int(string_interval)
        except ValueError:
            time_interval = string_interval.lower()
        if time_interval not in intervals:
            intervals.append(time_interval)
    return intervals


def __assert_intervals(input_string, bar_type="time"):
    """ Raise ValueError if there are invalid intervals in `input_string` """
    time_steps_set = set([1, 5, 10, 15, 30, 60, "day"])
    steps = extract_intervals_list(input_string)
    if bar_type == "time" and not set(steps).issubset(time_steps_set):
        raise ValueError(
            "There are invalid time intervals in '{}'".format(input_string)
        )
    if bar_type != "time" and not all([isinstance(s, int) for s in steps]):
        raise ValueError(
            "Invalid step for non-time bars in '{}'".format(input_string)
        )


def __remove_repetitions(input_string):
    """
    Remove any duplicated feature of `input_string` and sort parameters
    """
    key_value_list_split = input_string.split(":")
    keys = key_value_list_split[::2]
    str_vals = key_value_list_split[1::2]
    no_rep_map = {}
    for key, str_val in zip(keys, str_vals):
        if str_val[0] == "(":
            if key in no_rep_map.keys():
                no_rep_map[key] = no_rep_map[key] + "/" + str_val
            else:
                no_rep_map[key] = str_val
            continue
        if key in no_rep_map.keys() and str_val:
            raw_str_val = no_rep_map[key] + "," + str_val
        else:
            raw_str_val = str_val
        try:
            num_vals = list(map(int, set(raw_str_val.split(","))))
        except Exception:
            num_vals = list(map(float, set(raw_str_val.split(","))))
        num_vals.sort()
        str_val_unique = ",".join(map(str, num_vals))
        no_rep_map[key] = str_val_unique
    no_rep_list = []
    for no_rep_key, no_rep_val in no_rep_map.items():
        no_rep_list.append(no_rep_key)
        no_rep_list.append(no_rep_val)
    no_rep_input_string = ":".join(no_rep_list)
    return no_rep_input_string
