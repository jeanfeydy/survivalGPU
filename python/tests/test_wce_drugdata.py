import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import conversion, pandas2ri
from survivalgpu import WCESurvivalAnalysis

WCE = rpackages.importr("WCE")
ro.r("data('drugdata')")
drugdata_r = ro.r["drugdata"]


conv = conversion.get_conversion()
with conversion.localconverter(conv + pandas2ri.converter):
    drugdata_df = pandas2ri.rpy2py(drugdata_r)


#

model_survivalgpu_1knot = WCESurvivalAnalysis(
    cutoff=180, constrained="right", n_knots=1
)

model_survivalgpu_1knot.fit(
    dose=np.array(drugdata_df["dose"], dtype=np.float64),
    stop=np.array(drugdata_df["Stop"], dtype=np.int64),
    start=np.array(drugdata_df["Start"], dtype=np.int64),
    patient=np.array(drugdata_df["Id"], dtype=np.int64),
    event=np.array(drugdata_df["Event"], dtype=np.int64),
)
