import textwrap


def sanitise_name(name):
    """
    Function to sanitise model/model variant names to use in filenames (in Nextflow).
    """
    return name.replace(" ", "-")


def merge_dicts(d1, d2):
    """
    Merge two dictionaries recursively. d2 will overwrite d1 where specified.

    Assumes both dicts have same structure/keys.
    """
    for k, v in d2.items():
        if isinstance(v, dict):
            d1[k] = merge_dicts(d1[k], v)
        else:
            d1[k] = v
    return d1


def format_tooltip(text, width: int = 70):
    """
    Function to wrap text in a tooltip to the specified width. Ensures better-looking tooltips.

    Necessary because Qt only automatically wordwraps rich text, which has it's own issues.
    """
    return textwrap.fill(text.strip(), width=width, drop_whitespace=True)


def filter_empty_dict(d):
    """
    Filter out empty dicts from a nested dict.
    """
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            v = filter_empty_dict(v)
        if v not in (None, {}):
            new_dict[k] = v
    return new_dict
