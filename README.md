# interf-data-analysis

Repo for end-to-end data pipeline for radio interferometric data (primarily from ALMA), including integrating datasets, cleaning, characterizing noise levels, and performing signal extractions to measure emission lines.

Files:

- `match_filter_line.py`: pipeline and useful functions to detect spectral line signals and measure significance using a Gaussian matched-filtering scheme.
- `noise.py`: functions for measuring 1D and 2D rms values in data cubes
- `array_utils.py`: functions for working with ndarrays
- `redshift_utils.py`: functions for identifying spectral line transitions, converting frequencies and velocities, etc.
- `alma.py`: functions and constants specific to ALMA... hope to add other telescopes eventually.


Example line detection:

<img src="docs/images/example_line.png" height="300px" title="example line detection"/>

Notes:
 - This code is built on a [matched-filtering algorithm](https://en.wikipedia.org/wiki/Matched_filter), which presumes that lines take on a Gaussian profile. In theory, using a well-matched profile to the actual line shape will give the optimal S/N value. This is accomplished by convolving the dataset with multiple filter sizes in the spectral dimension.
 - This project is still under development, but the basic functionality already exists. Feel free to reach out to me with questions.
