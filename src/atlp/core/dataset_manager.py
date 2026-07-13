"""
    dataset_manager module is a collection of helper functions for managing the teleoperation dataset.

    It relies on atlp.core.interface (and possibly atlp.core.teleop_reader, if required)

    User is encouraged to extend this modulle if they want to add more quality-of-life functions
    without changing the core interface and teleop reader codes.
"""

from .interface import (
    label_field_values,
    vision_field_values,
    motion_field_values,
    get_datapoint,
    modify_datapoint,
    load_data,
    write_data,
    tag_get,
)

def _is_modelled(
    datapoint: str,
    use_dict: dict = dict(),
) -> bool:
    """
        Determine whether the datapoint has been modelled.
        (has the program automatically editted the value of the datapoint or not)

        Change the logic in this function if you want different way to check if the datapoint has been modelled.

        Args:
            datapoint: The name of the datapoint you want to check
            use_dict: (Optional) The dict of the header.json
                If supplied, the values will be retreived from the dict instead of the header file.

        Returns:
            True if the datapoint has been modelled, False otherwise.
    """
    datapoint_tags = get_single_datapoint(
        datapoint,
        main_field="label",
        field="tags",
        use_dict=use_dict,
    )
    if len(datapoint_tags) == 0: return False
    return True

def _is_checked(
    datapoint: str,
    use_dict: dict = dict(),
) -> bool:
    """
        Determine whether the datapoint has been checked.
        (user manually edits the values of the datapoint)

        Change the logic in this function if you want different way to check if the datapoint has been checked.

        Args:
            datapoint: The name of the datapoint you want to check
            use_dict: (Optional) The dict of the header.json
                If supplied, the values will be retreived from the dict instead of the header file.

        Returns:
            True if the datapoint has been checked, False otherwise.
    """
    datapoint_tags = get_single_datapoint(
        datapoint,
        main_field="label",
        field="tags",
        use_dict=use_dict,
        from_actual=True,
    )
    if len(datapoint_tags) == 0: return False
    return True

def get_single_datapoint(
    datapoint: str,
    main_field: str,
    field: str,
    from_actual: bool = False,
    use_dict: dict = dict(),
):
    """
        Get the value of a datapoint from a single field.

        Raise exception for invalid main_field and field.

        Return None if the value is not a valid value of that field
        (see _is_valid_field_value() for which value is considered valid)

        Args:
            datapoint: The name of the datapoint
            main_field: The name of the main_field. Must be either "label", "vision", or "motion"
            field: The name of the field
            from_actual: (Optional) If True, the value is retrieved from the actual field
            use_dict: (Optional) If True, the supplied dict will be used to retreive the value instead of the header file.

        Returns:
            The requested value of the datapoint from the specified main_field and field.
    """
    if main_field not in ["label", "vision", "motion"]:
        raise ValueError(f"Invalid main field. There is no {main_field}.")

    is_invalid_field = False

    if main_field == "label" and field not in label_field_values: is_invalid_field = True
    elif main_field == "vision" and field not in vision_field_values: is_invalid_field = True
    elif main_field == "motion" and field not in motion_field_values: is_invalid_field = True

    if is_invalid_field:
        raise ValueError(f"Invalid field. There is no {field} in {main_field}")

    labels = []
    visions = []
    motions = []

    if main_field == "label": labels = [field]
    elif main_field == "vision": visions = [field]
    elif main_field == "motion": motions = [field]

    data_from_get_datapoint = get_datapoint(
        datapoint,
        from_actual=from_actual,
        use_dict=use_dict,
        labels=labels,
        visions=visions,
        motions=motions,    
    )
    wanted_data = data_from_get_datapoint[main_field][field]
    return wanted_data

def filter(
  use_dict: dict = dict(),
  is_checked: bool | None = None,
  is_modelled: bool | None = None,
  # is_inconsistent = None,
  model_tag: str | None = None,
  actual_tag: str | None = None,
  model_is_failed: bool | None = None,
  actual_is_failed: bool | None = None,  
  model_is_self_collide: bool | None = None,
) -> list[str]:
    """
        Filter the datapoints to the specified keyword arguments (all optional)

        If the kwarg is None, the filter is not applied for that kwarg.

        Args:
            use_dict: The dictionary of the header.json.
                If supplied, the datapoint names will be retreieved from the dict instead of the header file.
            is_checked: True, False, or None
                The logic behind is_checked is inside _is_checked() (can be editted)
            is_modelled: True, False, or None
                The logic behind is_modelled is inside _is_modelled() (can be editted)
            model_tag: A string, or None 
            actual_tag: A string, or None
            model_is_failed: True, False, or None
            actual_is_failed: True, False, or None
            model_is_self_collide: True, False, or None

        Returns:
            List of strings of the datapoint's name which satisfied the filter conditions
    """
    if len(use_dict)==0: data=load_data()
    else: data=use_dict
        
    datapoint_set = set(data["datapoints"].keys())
    to_be_deleted_datapoint_set = set() 
    
    tag_list = tag_get(use_dict=data, from_actual=False)
    actual_tag_list = tag_get(use_dict=data, from_actual=True)

    filter_model_tag = True
    filter_actual_tag = True
    
    try:
        if str(model_tag) not in tag_list: filter_model_tag = False
    except Exception: filter_model_tag = False
                
    try:
        if str(actual_tag) not in actual_tag_list: filter_actual_tag = False
    except Exception: filter_actual_tag = False    

    if filter_model_tag:
        model_tag = str(model_tag)
        for datapoint in datapoint_set:
            datapoint_tags = get_single_datapoint(
                datapoint,
                main_field="label",
                field="tags",
                use_dict=data,
            )
            if model_tag not in datapoint_tags:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()
    
    if filter_actual_tag:
        actual_tag = str(actual_tag)
        for datapoint in datapoint_set:
            datapoint_tags = get_single_datapoint(
                datapoint,
                main_field="label",
                field="tags",
                use_dict=data,
                from_actual=True,
            )
            if actual_tag not in datapoint_tags:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if model_is_self_collide in [True, False]:        
        for datapoint in datapoint_set:
            datapoint_is_failed = get_single_datapoint(
                datapoint,
                main_field="motion",
                field="is_self_collide",
                use_dict=data,
            )
            if datapoint_is_failed != model_is_self_collide:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if model_is_failed in [True, False]:
        for datapoint in datapoint_set:
            datapoint_is_failed = get_single_datapoint(
                datapoint,
                main_field="label",
                field="is_failed",
                use_dict=data,
            )
            if datapoint_is_failed != model_is_failed:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()
    
    if actual_is_failed in [True, False]:
        for datapoint in datapoint_set:
            datapoint_is_failed = get_single_datapoint(
                datapoint,
                main_field="label",
                field="is_failed",
                use_dict=data,
                from_actual=True,
            )
            if datapoint_is_failed != actual_is_failed:
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if is_modelled in [True, False]:
        for datapoint in datapoint_set:
            if is_modelled != _is_modelled(datapoint, use_dict=data):
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    if is_checked in [True, False]:
        for datapoint in datapoint_set:
            if is_checked != _is_checked(datapoint, use_dict=data):
                to_be_deleted_datapoint_set.add(datapoint)
        datapoint_set.difference_update(to_be_deleted_datapoint_set)
        to_be_deleted_datapoint_set = set()

    datapoint_list = list(datapoint_set)
    return datapoint_list   

def reset_all(modify_actual: bool = False) -> None:
    """
        Reset every value of every fields of all datapoints to their default values

        Args:
            modify_actual: If true, reset the values from the actual fields
                    
        Warning:
            When called, all datapoint information is deleted.
    """

    data = load_data()

    for datapoint in data["datapoints"]:
        modify_datapoint(
            datapoint,
            use_dict=data,
            reset_all=True,
            modify_actual=modify_actual
        )
        # data["datapoints"][datapoint]["instruction"] = ""
        # data["datapoints"][datapoint]["tags"] = []
        # data["datapoints"][datapoint]["is_failed"] = False
        # data["datapoints"][datapoint]["is_labelled"] = False
        # data["datapoints"][datapoint]["is_checked"] = False

    # data["count_labelled"] = 0
    # data["count_failed"] = 0
    # data["count_checked"] = 0
    write_data(data)
    print("All tracked filed have been set to unlabelled")


def display_instruction(from_actual: bool = False) -> None:
    """
        Print instruction for each datapoint

        Args:
            from_actual: If True, display the actual field values instead
    """

    data = load_data()

    for datapoint in data["datapoints"]:

        instruction = get_datapoint(
            datapoint,
            from_actual=from_actual,
            use_dict=data,
            labels=['instruction']
        )

        if len(instruction) != 0: print(f"{datapoint} : {instruction['label']['instruction']}")