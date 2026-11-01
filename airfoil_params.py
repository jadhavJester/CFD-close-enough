"""Helpers for NACA 4-digit airfoil labels used by the interactive wind tunnel.

These pure functions were extracted from the pygame event loop in ``main.py``
so the camber/position/thickness stepping logic is unit-testable. A label has
the form ``"NACA{m}{p}{t:02d}"`` where ``m`` is the camber digit (0-9),
``p`` is the camber-position digit (0-9) and ``t`` is the thickness digits
(01-40).
"""


def parse_naca(airfoil_str):
    """Parse a ``"NACA...."`` label into (camber, position, thickness) ints.

    Returns ``None`` when the label is not a valid 4-digit NACA string
    (e.g. a KFM preset), so callers can leave other labels untouched.
    """
    if airfoil_str.startswith("NACA") and len(airfoil_str) == 8:
        try:
            return int(airfoil_str[4]), int(airfoil_str[5]), int(airfoil_str[6:8])
        except ValueError:
            return None
    return None


def build_naca(m, p, t):
    """Rebuild a NACA label from its camber, position and thickness digits."""
    return f"NACA{m}{p}{t:02d}"


def step_naca_param(airfoil_str, key):
    """Apply one Q/A/W/S/E/D keyboard step to a NACA label.

    Returns the updated label, or the input unchanged when it is not a
    NACA label. Digit ranges are clamped to the values ``main.py`` supports:
    camber 0-9, position 0-9, thickness 1-40.
    """
    params = parse_naca(airfoil_str)
    if params is None:
        return airfoil_str
    m, p, t = params
    key = key.lower()
    if key == "q":      # increase camber
        m = min(m + 1, 9)
    elif key == "a":    # decrease camber
        m = max(m - 1, 0)
    elif key == "w":    # move camber position forward
        p = min(p + 1, 9)
    elif key == "s":    # move camber position backward
        p = max(p - 1, 0)
    elif key == "e":    # increase thickness
        t = min(t + 1, 40)
    elif key == "d":    # decrease thickness
        t = max(t - 1, 1)
    return build_naca(m, p, t)
