# interf-data-analysis

Repo for working with radio interferometric data (primarily from ALMA), characterizing noise levels, and performing signal extractions to measure emission lines.

Contains:

- `match_filter_line.py`: pipeline and useful functions to detect spectral line signals and measure significance using a Gaussian matched-filtering scheme.
- `noise.py`: functions for measuring 1D and 2D rms values in data cubes
- `array_utils.py`: functions for working with ndarrays
- `redshift_utils.py`: functions for identifying spectral line transitions, converting frequencies and velocities, etc.
- `alma.py`: functions and constants specific to ALMA... hope to add other telescopes eventually.


Example line detection:

<img src="docs/images/example_line.png" height="300px" title="example line detection"/>
