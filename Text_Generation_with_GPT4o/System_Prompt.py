
def construct_prompt(
        length: str,
        vibe: str,
        style: str,
        mode: Literal["im-only", "txt-grounded", "im-txt-grounded"] = "im-only",
) -> str:
    length_map = {
        "short": "between 5 and 10 words",
        "medium": "between 10 and 20 words",
        "long": "between 20 and 40 words",
    }
    mode_task_map = {
        "im-only": "images",
        "txt-grounded": "technical data about bicycles",
        "im-txt-grounded": "images and technical data about bicycles contained in them",
    }

    image_wrapper = 'Images will be wrapped between <image i></image i> tags.\n' if 'im' in mode else ''
    bike_wrapper = 'Bike data will be wrapped between <data i></data i> tags.\n' if 'txt' in mode else ''
    description_usage = (
        'You do not have to include all, or any, '
        'of the bike data in the description if it does not fit the style or vibe. '
        'It is important that the description fits the constraints mentioned above.\n'
        if "txt" in mode else ''
    )

    prompt = (
        "Your task is to create descriptions of bicycles based on {0}. ".format(mode_task_map[mode])
        + "Each description should fulfill the following constraints: \n"
        + "- The length of the provided description should be {0}. \n".format(length_map[length])
        + "- The descriptions should be {0}. \n".format(vibe)
        + "- The descriptions should be in the style of a {0}. \n".format(style)
        + "{0}".format(image_wrapper)
        + "{0}".format(bike_wrapper)
        + "Wrap the resulting bike description in <description i></descriptions i> tags.\n"
        + "There should be *no* newlines in the descriptions.\n"
        + "{0}".format(description_usage)
        + "The descriptions should be very diverse within the given constraints."
    )

    return prompt