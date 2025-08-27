import numpy as np
import torch
from matplotlib import pyplot as plt
from survivalgpu import device, float32, int32
from survivalgpu.utils import numpy
from survivalgpu.wce_features import bspline_atoms, wce_features_batch

if False:
    import pykeops

    pykeops.clean_pykeops()

if __name__ == "__main__":
    # Sanity check 1: display the BSpline atoms

    order = 3  # 3 -> cubic splines
    nknots = 1
    cutoff = 20

    atoms, knots = bspline_atoms(nknots=nknots, cutoff=cutoff, order=order)
    print("Knots:", knots)
    print(atoms)

    plt.figure(figsize=(16, 10))
    for i, atom in enumerate(atoms.T):
        plt.plot(atom, label=f"BSpline {i}")

    plt.plot(torch.sum(atoms, axis=1), label="Sum")
    plt.xlabel("Time")
    plt.title(
        f"B-Spline atoms for cutoff={cutoff}, order={order}, nknots={nknots}"
    )
    plt.legend()

    plt.savefig("output_atoms.png")

    # Parameters of our BSpline window:
    order = 3  # 3 -> cubic splines
    nknots = 1
    cutoff = 20

    # Sampling times:
    times = torch.arange(-5, cutoff + 10, device=device, dtype=int32)
    N = len(times)

    # We study two patients on the same time-scale:
    times = torch.cat((times, times))  # (2*N,)

    # Ids to distinguish between patient 0 and patient 1:
    ids = torch.cat(
        (
            torch.zeros(N, device=device, dtype=int32),
            torch.ones(N, device=device, dtype=int32),
        )
    )

    # Doses:
    doses = torch.zeros(2 * N, device=device, dtype=float32)
    doses[(times == 0) & (ids == 0)] = 1

    doses[(times == 5) & (ids == 1)] = 1
    doses[(times == 10) & (ids == 1)] = 2

    features, knots = wce_features_batch(
        ids=ids,
        times=times,
        doses=doses,
        nknots=nknots,
        cutoff=cutoff,
        order=3,
    )

    print("Knots:", knots)

    # Fancy display:
    features = numpy(features)
    times = numpy(times)

    times = [times[:N], times[N:]]
    features = [features[:N], features[N:]]

    plt.figure(figsize=(16, 10))

    for patient in [0, 1]:
        plt.subplot(2, 1, patient + 1)
        for i, f in enumerate(features[patient].T):
            plt.plot(times[patient], f, label=f"BSpline {i}")
        plt.plot(
            times[patient], np.sum(features[patient], axis=1), label="Sum"
        )
        plt.xlabel("Time")
        plt.title(f"Patient {patient}")
        plt.legend()

    plt.savefig("output.png")
