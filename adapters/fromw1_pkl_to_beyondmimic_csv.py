import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


def from_amass_to_csv(amass_file: dict, scale: float = 0.6):
    if "motion 0" in amass_file:
        amass_file = amass_file["motion 0"]

    assert (
        "dof" in amass_file or "dof_pos" in amass_file
    ), "amass_file must contain 'dof' key"
    assert (
        "trans" in amass_file or "root_trans_offset" in amass_file
    ), "amass_file must contain 'trans' key"
    assert "root_rot" in amass_file, "amass_file must contain 'root_orient' key"

    if "dof" in amass_file:
        dof = amass_file["dof"]
    else:
        dof = amass_file["dof_pos"]

    if "trans" in amass_file:
        trans = amass_file["trans"]
    else:
        trans = amass_file["root_trans_offset"]

    trans[:, :2] -= trans[0:1, :2]
    trans[:, :2] *= scale
    trans[:, 2] += 0.15

    root_rot = amass_file["root_rot"]
    root_rot = R.from_quat(root_rot).as_quat()

    data = np.concatenate([trans, root_rot, dof], axis=1)
    return data


if __name__ == "__main__":
    import os

    inputdir = os.path.join(
        os.path.dirname(__file__), "..", "retarget", "data", "output"
    )
    outdir = os.path.join(os.path.dirname(__file__), "..", "data", "beyondmimic")
    for file in os.listdir(inputdir):
        if file.endswith(".pkl"):
            file_path = os.path.join(inputdir, file)
            data = joblib.load(file_path)
            data = from_amass_to_csv(data)

            output_file = os.path.join(outdir, file.replace(".pkl", ".csv"))
            pd.DataFrame(data).to_csv(output_file, index=False, header=False)
