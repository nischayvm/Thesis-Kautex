"""Identify the dominant render illumination from sampled RGB pixels."""

SAMPLE_STEP = 24


def detect_lighting(pixels):
    """Return a lighting label (or None) and the fractions of strong channel casts."""
    valid = 0
    red = green = chromatic = 0
    for r, g, b in pixels:
        high = max(r, g, b)
        if high < 24:
            continue
        valid += 1
        margin = max(12, high * 0.15)
        chromatic += high - min(r, g, b) >= margin
        red += r - max(g, b) >= margin
        green += g - max(r, b) >= margin
    if valid < 100:
        return None, (0.0, 0.0, 0.0)
    fractions = red / valid, green / valid, chromatic / valid
    if fractions[0] >= 0.30 and fractions[1] <= 0.10:
        return "red", fractions
    if fractions[1] >= 0.30 and fractions[0] <= 0.10:
        return "green", fractions
    if fractions[2] <= 0.08:
        return "regular", fractions
    return None, fractions
