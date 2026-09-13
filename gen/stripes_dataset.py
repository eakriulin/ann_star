import numpy as np
import matplotlib.pyplot as plt
import os

def generate_stripes_dataset(
    n_classes: int = 2,
    n_samples: int = 200,
    weights: list[float] | None = None,
    stripe_spacing: float = 1.0,
    stripe_spread: float = 0.3,
    stripes_per_class: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not (1 <= n_classes <= 20):
        raise ValueError("n_classes must be between 1 and 20")

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    if len(weights) != n_classes:
        raise ValueError("weights length must equal n_classes")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1.0")

    rng = np.random.default_rng(seed)

    total_stripes = n_classes * stripes_per_class
    # Center all stripes around x=0
    stripe_centers = np.array([
        (i - (total_stripes - 1) / 2.0) * stripe_spacing
        for i in range(total_stripes)
    ])

    X_list = []
    y_list = []

    assigned = 0
    for c in range(n_classes):
        n_c = int(round(n_samples * weights[c]))
        if c == n_classes - 1:
            # Last class gets remainder to hit exact total
            n_c = n_samples - assigned

        # Stripes for this class
        stripe_indices = list(range(c, total_stripes, n_classes))
        n_stripes = len(stripe_indices)

        # Distribute samples across stripes
        base = n_c // n_stripes
        rem = n_c % n_stripes
        counts = [base + 1] * rem + [base] * (n_stripes - rem)

        for idx, count in zip(stripe_indices, counts):
            if count <= 0:
                continue
            cx = stripe_centers[idx]
            x_coords = rng.normal(cx, stripe_spread, size=count)
            y_coords = rng.normal(0.0, stripe_spread, size=count)
            X_list.append(np.column_stack((x_coords, y_coords)))
            y_list.append(np.full(count, c, dtype=int))

        assigned += n_c

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Shuffle
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y

def save_stripes_dataset(
    generator='v1',
    n_classes: int = 2,
    n_samples: int = 200,
    weights: list[float] | None = None,
    stripe_spacing: float = 1.0,
    stripe_spread: float = 0.3,
    stripes_per_class: int = 2,
    seed: int | None = None,
    out_dir: str = "datasets/stripes",
    out_name: str = "stripes.data",
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, out_name)

    X, y = generate_stripes_dataset(
        n_classes=n_classes,
        n_samples=n_samples,
        weights=weights,
        stripe_spacing=stripe_spacing,
        stripe_spread=stripe_spread,
        stripes_per_class=stripes_per_class,
        seed=seed,
    )

    data = np.column_stack((X, y.astype(float)))
    np.savetxt(path, data, delimiter=",", fmt="%.5f")

    print(f"Saved {len(y)} samples to {path}")
    return path


if __name__ == "__main__":
    # EDIT PARAMS HERE
    N_CLASSES = 5
    N_SAMPLES = 1000
    STRIPE_SPACING = 1.0
    STRIPE_SPREAD = 0.25
    STRIPES_PER_CLASS = 2
    SEED = 42

    X, y = generate_stripes_dataset(
        n_classes=N_CLASSES,
        n_samples=N_SAMPLES,
        stripe_spacing=STRIPE_SPACING,
        stripe_spread=STRIPE_SPREAD,
        stripes_per_class=STRIPES_PER_CLASS,
        seed=SEED,
    )

    cmap = plt.colormaps["tab20"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for c in range(N_CLASSES):
        ax.scatter(X[y == c, 0], X[y == c, 1],
                   color=cmap(c), label=str(c), alpha=0.7, s=30)
    ax.set_title(f"stripes — {N_CLASSES} classes")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    os.makedirs("assets/imgs", exist_ok=True)
    fname = "assets/imgs/stripes_dataset.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f"Saved plot to {fname}")
