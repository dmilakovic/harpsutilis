import numpy as np
from typing import Tuple, Union, Dict, Any, Optional

# --- Configuration ---
# Defines properties for different file types (ftype)
# - default: The default integer version number.
# - keys: Tuple of keys used when version is specified as a dictionary.
# - val_ranges: Optional tuple of (min, max) tuples for each key.
VERSION_CONFIG = {
    'wavesol': {
        'default': 701,  # e.g., polyord=7, gaps=0, segment=1
        'keys': ('polyord', 'gaps', 'segment'),
        # Example ranges (val1, val2, val3)
        'val_ranges': ((0, 99), (0, 9), (0, 9))
    },
    'lsf': {
        'default': 111,  # e.g., iteration=1, model_scatter=1, interpolate=1
        'keys': ('iteration', 'model_scatter', 'interpolate'),
        'val_ranges': ((0, 99), (0, 9), (0, 9))
    }
}
# Versions below this are considered "simple" and not packed from components.
MIN_PACKED_VERSION_THRESHOLD = 100 # Versions >= 100 are potentially packed

def _get_ftype_config(ftype: str) -> Dict[str, Any]:
    """Helper to get configuration for a given ftype."""
    if ftype not in VERSION_CONFIG:
        raise ValueError(f"Unknown ftype: '{ftype}'. Supported ftypes are: {list(VERSION_CONFIG.keys())}")
    return VERSION_CONFIG[ftype]

def get_default_version_int(ftype: str) -> int:
    """Returns the default integer version for the given ftype."""
    config = _get_ftype_config(ftype)
    return config['default']

def unpack_integer(version_int: int, ftype: str) -> Tuple[int, int, int]:
    """
    Unpacks a 3 or 4-digit integer version into its three component values.
    Assumes version_int >= MIN_PACKED_VERSION_THRESHOLD.

    Example:
    701 (for wavesol: polyord=7, gaps=0, segment=1) -> (7, 0, 1)
    1001 (for wavesol: polyord=10, gaps=0, segment=1) -> (10, 0, 1)

    Args:
        version_int (int): The packed integer version (e.g., 701, 1001).
        ftype (str): The file type ('wavesol', 'lsf'), used for validation.

    Returns:
        Tuple[int, int, int]: The three unpacked integer components (val1, val2, val3).
    """
    if not isinstance(version_int, (int, np.integer)):
        raise TypeError(f"Version to unpack must be an integer, got {type(version_int)}.")
    if version_int < MIN_PACKED_VERSION_THRESHOLD:
        raise ValueError(
            f"Cannot unpack version {version_int} using component logic. "
            f"It's below threshold {MIN_PACKED_VERSION_THRESHOLD}."
        )

    s_ver = str(version_int)
    num_digits = len(s_ver)

    if num_digits == 3: # e.g., 701 -> val1=7, val2=0, val3=1
        val1 = int(s_ver[0])
        val2 = int(s_ver[1])
        val3 = int(s_ver[2])
    elif num_digits == 4: # e.g., 1001 -> val1=10, val2=0, val3=1 (from format val1:02d)
        val1 = int(s_ver[0:2]) # First two digits
        val2 = int(s_ver[2])   # Third digit
        val3 = int(s_ver[3])   # Fourth digit
    else:
        raise ValueError(
            f"Packed version {version_int} must be 3 or 4 digits long for unpacking, "
            f"based on val1 (1-2 digits), val2 (1 digit), val3 (1 digit) scheme. Got {num_digits} digits."
        )

    # Validate against configured ranges
    config = _get_ftype_config(ftype)
    ranges = config.get('val_ranges')
    if ranges:
        vals = (val1, val2, val3)
        for i, val_comp in enumerate(vals):
            min_val, max_val = ranges[i]
            if not (min_val <= val_comp <= max_val):
                key_name = config['keys'][i]
                raise ValueError(
                    f"Unpacked component '{key_name}' ({val_comp}) for ftype '{ftype}' "
                    f"is out of allowed range [{min_val}, {max_val}] for version {version_int}."
                )
    return val1, val2, val3

def unpack_dictionary(item_dict: Dict[str, int], ftype: str) -> Tuple[int, int, int]:
    """
    Extracts three component values from a dictionary based on ftype.
    Uses default component values if keys are missing.
    """
    config = _get_ftype_config(ftype)
    keys = config['keys']
    default_version_int = config['default']

    # Get default components by unpacking the default integer version
    # This ensures defaults are consistent with the packing scheme
    default_val1, default_val2, default_val3 = unpack_integer(default_version_int, ftype)
    default_components = (default_val1, default_val2, default_val3)

    val1 = int(item_dict.get(keys[0], default_components[0]))
    val2 = int(item_dict.get(keys[1], default_components[1]))
    val3 = int(item_dict.get(keys[2], default_components[2]))
    
    # Validate against configured ranges
    ranges = config.get('val_ranges')
    if ranges:
        vals = (val1, val2, val3)
        for i, val_comp in enumerate(vals):
            min_val, max_val = ranges[i]
            if not (min_val <= val_comp <= max_val):
                key_name = config['keys'][i]
                raise ValueError(
                    f"Component '{key_name}' ({val_comp}) from dictionary for ftype '{ftype}' "
                    f"is out of allowed range [{min_val}, {max_val}]."
                )
    return val1, val2, val3

def item_to_version(item: Optional[Union[int, Dict, Tuple]], ftype: str) -> int:
    """
    Converts an item representation into a canonical integer version.

    - If item is None, returns the default version for the ftype.
    - If item is an int < MIN_PACKED_VERSION_THRESHOLD, it's a "simple" version and returned as is.
    - If item is an int >= MIN_PACKED_VERSION_THRESHOLD, it's treated as a "packed"
      version, unpacked, validated, and repacked into canonical form.
    - If item is a dict, its components are extracted and packed.
    - If item is a tuple (val1, val2, val3), its components are packed.

    The packed version format is val1 (2 digits, zero-padded), val2 (1 digit), val3 (1 digit).
    Example: (7,0,1) -> 0701, (10,0,1) -> 1001.

    Args:
        item: The version representation.
        ftype (str): The file type ('wavesol', 'lsf'), crucial for defaults and dict keys.

    Returns:
        int: The canonical integer version.
    """
    config = _get_ftype_config(ftype) # Validates ftype early

    if item is None:
        return config['default']

    val1, val2, val3 = -1, -1, -1 # Initialize to invalid values

    if isinstance(item, dict):
        val1, val2, val3 = unpack_dictionary(item, ftype)
    elif isinstance(item, (int, np.integer)):
        if item < MIN_PACKED_VERSION_THRESHOLD:
            return int(item)  # Simple version, return as is
        else:
            # Packed version, unpack to validate and get components, then repack
            val1, val2, val3 = unpack_integer(int(item), ftype)
    elif isinstance(item, tuple):
        if len(item) == 3 and all(isinstance(x, (int, np.integer)) for x in item):
            val1, val2, val3 = int(item[0]), int(item[1]), int(item[2])
            # Validate tuple components against ranges
            ranges = config.get('val_ranges')
            if ranges:
                vals_tuple = (val1, val2, val3)
                for i, val_comp in enumerate(vals_tuple):
                    min_val, max_val = ranges[i]
                    if not (min_val <= val_comp <= max_val):
                        key_name = config['keys'][i]
                        raise ValueError(
                            f"Component '{key_name}' ({val_comp}) from tuple for ftype '{ftype}' "
                            f"is out of allowed range [{min_val}, {max_val}]."
                        )
        else:
            raise ValueError(f"Version tuple must be (int, int, int), got {item}")
    else:
        raise TypeError(f"Unsupported type for version item: {type(item)}")

    # Pack into canonical integer form (val1:02d, val2:1d, val3:1d)
    # This ensures, for example, (7,0,1) and an input of 701 both result in the same canonical int.
    # Using :02d means val1=7 becomes "07", so 7,0,1 -> "0701" -> 701.
    # And val1=10 becomes "10", so 10,0,1 -> "1001" -> 1001.
    return int(f"{val1:02d}{val2:1d}{val3:1d}")


def extract_item(item_input: Any, ftype: str) -> Tuple[Any, Optional[int], bool]:
    """
    Utility function to extract an "item" (extension name/number) and its version.

    Args:
        item_input: Can be:
            - A simple value (e.g., 'EXTNAME', 1): Treated as extension, version is default.
            - A 1-tuple (e.g., ('EXTNAME',)): Extension from tuple, version is default.
            - A 2-tuple (e.g., ('EXTNAME', version_spec)): Extension and version_spec.
              `version_spec` can be int, dict, tuple, or None (resolves to default).
        ftype (str): The file type ('wavesol', 'lsf') to guide version processing.

    Returns:
        Tuple[Any, Optional[int], bool]: (extension, version_int, version_explicitly_sent)
            - extension: The item identifier.
            - version_int: The canonical integer version, or None if not resolved from input.
            - version_explicitly_sent: True if version information was part of a tuple input.
    """
    ext: Any
    version_val: Optional[Union[int, Dict, Tuple]] = None # Raw version from input
    version_explicitly_sent: bool = False

    if isinstance(item_input, tuple):
        version_explicitly_sent = True # A tuple implies intent regarding version
        num_elements = len(item_input)
        if num_elements == 1:
            ext = item_input[0]
            # version_val remains None, item_to_version will use default for ftype
        elif num_elements == 2:
            ext, version_val = item_input
            # version_val can be int, dict, tuple, or None.
            # item_to_version will handle these, including using default if version_val is None.
        else:
            raise ValueError(f"Item tuple must have 1 or 2 elements, got {num_elements}")
    else:
        ext = item_input
        # version_val remains None. If we want a default version even for non-tuple inputs,
        # we would set version_val to None here too, letting item_to_version pick it up.
        # However, current logic: if not a tuple, version is not considered "sent" or processed
        # unless explicitly done by the caller with the result.
        # For consistency with original `extract_item` where `ver=None` if `item` wasn't a tuple,
        # we will NOT automatically assign a default here. A default is only applied if a
        # version slot was provided (e.g. `('EXT', None)`).
        # So, if no version info is in item_input, resolved_version will be None.
        if version_explicitly_sent: # This branch won't be hit due to outer if
             resolved_version = item_to_version(version_val, ftype)
        else:
             resolved_version = None # Keep it None if not explicitly sent via tuple
        return ext, resolved_version, version_explicitly_sent


    # If version_val is None (from 1-tuple or 2-tuple with None), item_to_version gets default.
    # If version_val has a value, item_to_version processes it.
    resolved_version = item_to_version(version_val, ftype)

    return ext, resolved_version, version_explicitly_sent


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Wavesol Examples ---")
    ft = 'wavesol'
    # Default
    ext, ver, sent = extract_item('SCI', ftype=ft)
    print(f"Item: 'SCI' -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: None for ver unless extract_item changes)")
    # Corrected expectation for extract_item based on its refined logic:
    # if 'SCI' is passed, version_val is None, and version_explicitly_sent is False.
    # The current extract_item returns None for version if not explicitly sent.
    # To get default for 'SCI', you'd call item_to_version(None, ft) separately or change extract_item.
    # Let's refine extract_item to assign default if only ext is given. No, let's stick to a more direct interpretation.
    # If user provides 'SCI', they didn't ask for a version.
    # If they provide ('SCI', None), they asked for "the version for SCI", which means default.

    ext, ver, sent = extract_item(('SCI', None), ftype=ft) # Version explicitly requested, resolves to default
    print(f"Item: ('SCI', None) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: {get_default_version_int(ft)})")

    ext, ver, sent = extract_item(('SCI', 701), ftype=ft)
    print(f"Item: ('SCI', 701) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 701)")

    ext, ver, sent = extract_item(('SCI', 1001), ftype=ft) # polyord=10, gaps=0, segment=1
    print(f"Item: ('SCI', 1001) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 1001)")

    ext, ver, sent = extract_item(('SCI', {'polyord': 8}), ftype=ft) # gaps=0, segment=1 from default 701
    print(f"Item: ('SCI', {{'polyord': 8}}) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 801)")

    ext, ver, sent = extract_item(('SCI', (12, 1, 0)), ftype=ft) # polyord=12, gaps=1, segment=0
    print(f"Item: ('SCI', (12,1,0)) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 1210)")

    ext, ver, sent = extract_item(('SCI', 1), ftype=ft) # Simple version
    print(f"Item: ('SCI', 1) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 1)")

    print(f"\nUnpacking 701 for {ft}: {unpack_integer(701, ft)}")
    print(f"Unpacking 1001 for {ft}: {unpack_integer(1001, ft)}")
    print(f"Unpacking 1210 for {ft}: {unpack_integer(1210, ft)}")

    print("\n--- LSF Examples ---")
    ft_lsf = 'lsf'
    ext, ver, sent = extract_item(('MODEL', None), ftype=ft_lsf)
    print(f"Item: ('MODEL', None) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: {get_default_version_int(ft_lsf)})")

    ext, ver, sent = extract_item(('MODEL', {'iteration': 2, 'interpolate': 0}), ftype=ft_lsf)
    # model_scatter=1 from default 111
    print(f"Item: ('MODEL', {{'iter':2,'interp':0}}) -> ext: {ext}, ver: {ver} (sent: {sent}) (Expected: 210)")

    print(f"\nUnpacking 111 for {ft_lsf}: {unpack_integer(111, ft_lsf)}")
    print(f"Unpacking 210 for {ft_lsf}: {unpack_integer(210, ft_lsf)}")


    print("\n--- Error Handling Examples ---")
    try:
        item_to_version(99999, ft) # Too many digits for packing
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        item_to_version({'polyord': 100}, ft) # polyord out of 0-99 range
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        unpack_integer(50, ft) # Too small to unpack
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        extract_item(('SCI', (1,2,3,4)), ftype=ft) # Bad tuple
    except ValueError as e:
        print(f"Caught expected error: {e}")
    try:
        item_to_version("bad_string", ftype=ft) # Bad type
    except TypeError as e:
        print(f"Caught expected error: {e}")
    try:
        item_to_version(None, ftype="unknown_ftype")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Example of how extract_item handles non-tuple input
    ext_val, ver_val, sent_val = extract_item("JUST_EXT_NAME", ftype='wavesol')
    print(f"\nItem: 'JUST_EXT_NAME' -> ext: {ext_val}, ver: {ver_val} (sent: {sent_val})")
    # If you want default for "JUST_EXT_NAME", you need to call item_to_version explicitly:
    ver_for_just_ext = item_to_version(None, 'wavesol')
    print(f"Default version for 'wavesol' if only ext name given: {ver_for_just_ext}")