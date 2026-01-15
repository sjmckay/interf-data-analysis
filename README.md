# interf-data-analysis

Repo for working with radio interferometric data (primarily from ALMA), characterizing noise levels, and performing signal extractions to measure emission lines.

Contains:

    - `spectral_line.py`: pipeline and useful functions to ID peaks, measure spectral line significance, and compute errors using a Gaussian matched-filtering scheme.
    - `array_utils.py`: functions for working with ndarrays
    - `redshift_utils.py`: functions for ID'ing spectral lines, converting frequencies and velocities, etc.
    - `alma.py`: functions and constants specific to ALMA... hope to add other telescopes eventually.


Example line detection:

<img src="docs/images/example_line.png" height="300px" title="example line detection"/>
