"""DiverScan utilities.

Add the utils folder to ``sys.path`` and import the modules you need:

    import sys
    sys.path.insert(0, utils_folder)

    import learning_utils
    import preprocessing
    import postprocessing

Importing the package itself also works when its parent directory is on
``sys.path``:

    from utils import learning_utils

The modules are deliberately not re-exported here, so that importing the
package does not pull in heavy dependencies: ``postprocessing`` needs plotly,
folium and ipywidgets, while ``learning_utils``, ``single_model`` and
``multi_model`` need torch.
"""

__all__ = [
    "learning_utils",
    "multi_model",
    "single_model",
    "preprocessing",
    "postprocessing",
    "make_summary_table",
    "run_record",
]

__version__ = "1.0.0"
