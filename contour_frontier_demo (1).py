# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

def contour_to_frontiers_verbose(contour: np.ndarray, unexplored_mask: np.ndarray
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    num_contour_points = len(contour)
    print(f"[Step 1] Number of contour points: {num_contour_points}")

    bad_inds = []
    for idx in range(num_contour_points):
        x, y = contour[idx][0]
        if unexplored_mask[y, x] == 0:
            bad_inds.append(idx)
    print(f"[Step 2] bad_inds (mask==0 at these contour indices): {bad_inds}")

    frontiers_raw = np.split(contour, bad_inds) if len(bad_inds) > 0 else [contour]
    print(f"[Step 3] np.split produced {len(frontiers_raw)} raw piece(s):")
    for i, f in enumerate(frontiers_raw):
        start = f[0][0].tolist()
        end = f[-1][0].tolist()
        print(f"  - raw[{i}]: len={len(f)}, start={start}, end={end}")

    front_last_split = (
        (0 not in bad_inds)
        and (len(bad_inds) > 0)
        and (max(bad_inds) < num_contour_points - 2)
    )
    print(f"[Step 4] front_last_split detected? {front_last_split}")

    filtered_frontiers = []
    for idx, f in enumerate(frontiers_raw):
        if len(f) > 2 or (idx == 0 and front_last_split):
            if idx == 0:
                filtered_frontiers.append(f)
            else:
                filtered_frontiers.append(f[1:])

    print(f"[Step 5] After filtering/trim: {len(filtered_frontiers)} piece(s):")
    for i, f in enumerate(filtered_frontiers):
        print(f"  - filtered[{i}]: len={len(f)}")

    if len(filtered_frontiers) > 1 and front_last_split:
        last_frontier = filtered_frontiers.pop()
        filtered_frontiers[0] = np.concatenate((last_frontier, filtered_frontiers[0]))
        print("[Step 6] Merged last piece into the first (ring closure).")

    print(f"[Result] Final frontier segments: {len(filtered_frontiers)} piece(s).")
    for i, f in enumerate(filtered_frontiers):
        start = f[0][0].tolist()
        end = f[-1][0].tolist()
        print(f"  - frontier[{i}]: len={len(f)}, start={start}, end={end}")
    return filtered_frontiers, {"bad_inds": bad_inds}

def build_demo() -> Tuple[np.ndarray, np.ndarray, list]:
    H, W = 20, 20
    mask = np.ones((H, W), dtype=np.uint8)

    pts = []
    for x in range(4, 16):
        pts.append([x, 5])
    for y in range(6, 15):
        pts.append([15, y])
    for x in range(15, 3, -1):
        pts.append([x, 14])
    for y in range(14, 5, -1):
        pts.append([4, y])

    contour = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

    N = len(contour)
    bad_idx_candidates = [7, 18, N - 5]
    for idx in bad_idx_candidates:
        x, y = contour[idx, 0]
        mask[y, x] = 0

    return contour, mask, bad_idx_candidates

def show_mask(mask: np.ndarray, title: str = "Unexplored Mask (1=unexplored, 0=not)") -> None:
    plt.figure()
    plt.title(title)
    plt.imshow(mask)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def plot_contour(contour: np.ndarray, title: str = "Input Contour") -> None:
    xs = contour[:, 0, 0]
    ys = contour[:, 0, 1]
    plt.figure()
    plt.title(title)
    plt.plot(xs, ys, marker="o")
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

def plot_frontiers(frontiers: List[np.ndarray], title: str = "Final Frontiers") -> None:
    plt.figure()
    plt.title(title)
    for i, f in enumerate(frontiers):
        xs = f[:, 0, 0]
        ys = f[:, 0, 1]
        plt.plot(xs, ys, marker="o", label=f"frontier {i}")
    plt.gca().invert_yaxis()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    contour, mask, manual_bad = build_demo()
    print("\\n=== Demo Setup ===")
    print(f"Contour length: {len(contour)}")
    print(f"Manually chosen 'bad' indices for the demo: {manual_bad}\\n")

    show_mask(mask, "Unexplored Mask (demo)")
    plot_contour(contour, "Full Input Contour (demo)")

    frontiers, _ = contour_to_frontiers_verbose(contour, mask)
    plot_frontiers(frontiers, "Frontiers After Splitting & Merging (demo)")

if __name__ == "__main__":
    main()
