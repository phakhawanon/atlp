from ..core.interface import (
        get_datapoint,
        enable_actual_field,
        load_data,
    )
import numpy as np
from numpy.typing import NDArray

def evaluate_tag() -> NDArray[np.float64]:
    """
        Evaluate the accuracy of the ATLP by returning the confusion matrix

        Currently, only consider
            - datapoints that have been labelled by the model (indicated by non-empty 'model'/'label'/'instruction' field)
            - datapoints whose 'actual'/'label'/'tags' has only one tag

        Do nothing if enable_actual_field is False

        Returns:
            A confusion matrix of size (N+1, N) where N is the number of actual tags
            
            The rows represent the model field, and the columns represent the actual field.

            The first N rows and columns represent the tags, whose order are the same to the order received by tag_get()

            The last row represent either
                1. model tags that are not in the actual tag list
                2. more than one model tags for one datapoint
    """

    if not enable_actual_field: return

    data = load_data()
    n = 0
    # tag_list = data["tag_list"]
    actual_tag_list = data["actual_tag_list"]
    confusion_size = ( len(actual_tag_list)+1, len(actual_tag_list) )
    confusion = np.zeros(confusion_size, float)

    for datapoint in data["datapoints"]:

        tags = get_datapoint(datapoint, use_dict=data, labels=["tags"])["label"]["tags"]
        actual_tags_request = get_datapoint(datapoint, from_actual=True, use_dict=data, labels=["tags"])
        # print(actual_tags_request)

        if len(actual_tags_request) != 0:

            actual_tags = actual_tags_request["label"]["tags"]
            # print(actual_tags)
            
            if len(actual_tags) == 1:

                n += 1

                actual_tag = actual_tags[0]
                actual_tag_index = actual_tag_list.index(actual_tag)

                if len(tags) == 1:

                    tag = tags[0]

                    if tag in actual_tag_list:
                        
                        tag_index = actual_tag_list.index(tag)
                        confusion[tag_index, actual_tag_index] += 1

                    else:

                        confusion[len(actual_tag_list), actual_tag_index] += 1

                else: confusion[len(actual_tag_list), actual_tag_index] += 1

        # print(confusion)
        # print(n)
            
    return confusion / n
