import numpy as np
import phasepack

def phase_congruency_features(image):

    M, m = phasepack.phasecong(
        image,
        nscale=4,          # match your Log-Gabor
        norient=6,         # from paper
        covariance_only=True
    )

    return [
        np.mean(M),
        np.std(M),
        np.mean(m),
        np.std(m)
    ]

X_pc = []
y = []

for image, label in dataset:
    feat = phase_congruency_features(image)
    X_pc.append(feat)
    y.append(label)

X_pc = np.array(X_pc)

pc_feature_names = [
    "PC_mean_M",
    "PC_std_M",
    "PC_mean_m",
    "PC_std_m"
]