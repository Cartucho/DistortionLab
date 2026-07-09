from pathlib import Path

import cv2 as cv
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.linalg import lstsq


def diagnostics_dir(output_dir):
    path = Path(output_dir) / "visualizations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def draw_coverage_image(output_dir, pts_img, vid_width, vid_height, grid_size=16):
    """Draw and save selected-frame corner coverage."""
    grid = np.zeros((grid_size, grid_size), dtype=int)
    cell_width = vid_width / grid_size
    cell_height = vid_height / grid_size

    for corners in pts_img:
        for corner in corners:
            x, y = corner[0][0], corner[0][1]
            gx = min(int(x / cell_width), grid_size - 1)
            gy = min(int(y / cell_height), grid_size - 1)
            grid[gy, gx] += 1

    max_count = int(grid.max())
    all_points = [corners.reshape(-1, 2) for corners in pts_img]
    all_points = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 2))

    positive_levels = max(max_count, 1)
    green_colors = np.array([
        mcolors.to_rgba((0.16, 0.45 + 0.35 * (i / max(positive_levels - 1, 1)), 0.16))
        for i in range(positive_levels)
    ])
    colors = np.vstack([mcolors.to_rgba("#5b1111"), green_colors])
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, positive_levels + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    ax.imshow(
        grid,
        extent=[0, vid_width, vid_height, 0],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    for i in range(grid_size + 1):
        ax.axvline(i * cell_width, color="#707070", linewidth=0.8, alpha=0.7)
        ax.axhline(i * cell_height, color="#707070", linewidth=0.8, alpha=0.7)

    if all_points.size > 0:
        ax.scatter(
            all_points[:, 0], all_points[:, 1],
            s=10, c="white", edgecolors="black", linewidths=0.35, alpha=0.95,
        )

    ax.set_xlim(0, vid_width)
    ax.set_ylim(vid_height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Inlier Coverage: red = empty cell | brighter green = more detected corners | white dots = detected corners",
        color="white",
        fontsize=12,
        pad=10,
    )

    out_path = diagnostics_dir(output_dir) / "coverage.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_residual_grid(output_dir, pts_obj, pts_img, mtx, dist, rvecs, tvecs,
                       vid_width, vid_height, grid_size=16, arrow_scale=100.0):
    """Draw and save a grid of mean reprojection residuals."""
    positions = []
    residuals = []
    for objp, imgp, rvec, tvec in zip(pts_obj, pts_img, rvecs, tvecs):
        observed = imgp.reshape(-1, 2)
        projected, _ = cv.projectPoints(objp, rvec, tvec, mtx, dist)
        projected = projected.reshape(-1, 2)
        positions.append(observed)
        residuals.append(observed - projected)

    if not positions:
        return

    pos = np.concatenate(positions, axis=0)
    res = np.concatenate(residuals, axis=0)

    cell_width = vid_width / grid_size
    cell_height = vid_height / grid_size
    ix = np.clip((pos[:, 0] / cell_width).astype(int), 0, grid_size - 1)
    iy = np.clip((pos[:, 1] / cell_height).astype(int), 0, grid_size - 1)

    counts = np.zeros((grid_size, grid_size), dtype=int)
    mean_dx = np.zeros((grid_size, grid_size))
    mean_dy = np.zeros((grid_size, grid_size))
    mean_mag = np.full((grid_size, grid_size), np.nan)
    sum_mag = np.zeros((grid_size, grid_size))
    magnitudes = np.linalg.norm(res, axis=1)

    for i in range(pos.shape[0]):
        gx, gy = ix[i], iy[i]
        counts[gy, gx] += 1
        mean_dx[gy, gx] += res[i, 0]
        mean_dy[gy, gx] += res[i, 1]
        sum_mag[gy, gx] += magnitudes[i]

    mask = counts > 0
    mean_dx[mask] /= counts[mask]
    mean_dy[mask] /= counts[mask]
    mean_mag[mask] = sum_mag[mask] / counts[mask]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=0.02, right=0.90, top=0.98, bottom=0.08)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    im = ax.imshow(
        mean_mag,
        extent=[0, vid_width, vid_height, 0],
        cmap="plasma",
        interpolation="nearest",
        aspect="equal",
    )

    cx = (np.arange(grid_size) + 0.5) * cell_width
    cy = (np.arange(grid_size) + 0.5) * cell_height
    CX, CY = np.meshgrid(cx, cy)
    ax.quiver(
        CX[mask],
        CY[mask],
        mean_dx[mask] * arrow_scale,
        mean_dy[mask] * arrow_scale,
        color="white",
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.002,
        headwidth=3,
        headlength=3,
        headaxislength=2,
    )

    ax.set_xlim(0, vid_width)
    ax.set_ylim(vid_height, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean residual magnitude [px]", color="white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("white")
    fig.text(
        0.5,
        0.025,
        f"Cell color = mean reprojection error | arrows = mean residual direction ({arrow_scale:.0f}x)",
        ha="center",
        va="bottom",
        color="white",
        fontsize=11,
    )

    out_path = diagnostics_dir(output_dir) / "residual_grid.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_residual_normality(output_dir, pts_obj, pts_img, mtx, dist, rvecs, tvecs):
    """Draw and save a per-component residual normality check."""
    residuals = []
    for objp, imgp, rvec, tvec in zip(pts_obj, pts_img, rvecs, tvecs):
        observed = imgp.reshape(-1, 2)
        projected, _ = cv.projectPoints(objp, rvec, tvec, mtx, dist)
        projected = projected.reshape(-1, 2)
        residuals.append(observed - projected)

    if not residuals:
        return

    values = np.concatenate(residuals, axis=0).ravel()
    mu = float(np.median(values))
    mad = float(np.median(np.abs(values - mu)))
    sigma = 1.4826 * mad

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.12)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("white")

    counts, bins, _ = ax.hist(
        values, bins=60, color="#00d4ff", edgecolor="#111111", alpha=0.9
    )
    if sigma > 0:
        x = np.linspace(bins[0], bins[-1], 500)
        pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        bin_width = bins[1] - bins[0]
        ax.plot(
            x,
            pdf * len(values) * bin_width,
            color="#ffd84d",
            linewidth=2.0,
            label=f"normal fit (MAD sigma={sigma:.3f}px)",
        )
    ax.axvline(mu, color="white", linewidth=1.4, label=f"median {mu:.3f}px")
    ax.set_title(
        f"Residual Component Distribution (N={len(values)}, MAD sigma={sigma:.3f}px)",
        pad=14,
    )
    ax.set_xlabel("x/y residual component [px]")
    ax.set_ylabel("Count")
    ax.grid(True, linewidth=0.5, alpha=0.2, color="white")
    legend = ax.legend(facecolor="#111111", edgecolor="white", framealpha=0.85)
    for text in legend.get_texts():
        text.set_color("white")

    out_path = diagnostics_dir(output_dir) / "residual_normality.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_undistortion_grid(output_dir, image, K, D, max_preview_dim=1600, grid_cells=30):
    """Draw and save a distortion/undistortion preview using one calibration frame."""
    if image is None:
        return

    h, w = image.shape[:2]
    scale = min(1.0, max_preview_dim / max(w, h))
    if scale < 1.0:
        preview = cv.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv.INTER_AREA)
    else:
        preview = image.copy()

    ph, pw = preview.shape[:2]
    K_preview = np.array(K, dtype=np.float64).copy()
    K_preview[0, :] *= scale
    K_preview[1, :] *= scale

    mapx, mapy = cv.initUndistortRectifyMap(
        K_preview, D, None, K_preview, (pw, ph), cv.CV_32FC1
    )
    undistorted = cv.remap(preview, mapx, mapy, cv.INTER_LANCZOS4)

    cmap = plt.colormaps["jet"]
    grid_step = max(max(pw, ph) / grid_cells, 10.0)
    x_lines = np.arange(grid_step, pw, grid_step)
    y_lines = np.arange(grid_step, ph, grid_step)

    def colored_segments(xs, ys):
        pts = np.column_stack([xs, ys])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        diag = ((xs[:-1] + xs[1:]) / (2 * pw) + (ys[:-1] + ys[1:]) / (2 * ph)) / 2
        return segs, cmap(diag)

    def remap_points(points):
        x = np.clip(points[:, 0], 0, pw - 2)
        y = np.clip(points[:, 1], 0, ph - 2)
        ix = x.astype(int)
        iy = y.astype(int)
        fx = x - ix
        fy = y - iy
        dx = (
            (1 - fx) * (1 - fy) * mapx[iy, ix]
            + fx * (1 - fy) * mapx[iy, ix + 1]
            + (1 - fx) * fy * mapx[iy + 1, ix]
            + fx * fy * mapx[iy + 1, ix + 1]
        )
        dy = (
            (1 - fx) * (1 - fy) * mapy[iy, ix]
            + fx * (1 - fy) * mapy[iy, ix + 1]
            + (1 - fx) * fy * mapy[iy + 1, ix]
            + fx * fy * mapy[iy + 1, ix + 1]
        )
        return np.column_stack([dx, dy])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.patch.set_facecolor("#111111")
    for ax in axes.flat:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    n_samples = 256
    line_width = 1.0

    for x0 in x_lines:
        ys = np.linspace(0, ph - 1, n_samples)
        xs = np.full_like(ys, x0)
        segs, colors = colored_segments(xs, ys)
        axes[0, 1].add_collection(LineCollection(segs, colors=colors, linewidths=line_width))

        xi = int(round(x0))
        y_idx = np.linspace(0, ph - 1, n_samples).astype(int)
        dx = mapx[y_idx, xi]
        dy = mapy[y_idx, xi]
        mapped = np.stack([dx, dy], axis=1)
        mapped_segs = np.stack([mapped[:-1], mapped[1:]], axis=1)
        axes[0, 0].add_collection(LineCollection(mapped_segs, colors=colors, linewidths=line_width))

    for y0 in y_lines:
        xs = np.linspace(0, pw - 1, n_samples)
        ys = np.full_like(xs, y0)
        segs, colors = colored_segments(xs, ys)
        axes[0, 1].add_collection(LineCollection(segs, colors=colors, linewidths=line_width))

        yi = int(round(y0))
        x_idx = np.linspace(0, pw - 1, n_samples).astype(int)
        dx = mapx[yi, x_idx]
        dy = mapy[yi, x_idx]
        mapped = np.stack([dx, dy], axis=1)
        mapped_segs = np.stack([mapped[:-1], mapped[1:]], axis=1)
        axes[0, 0].add_collection(LineCollection(mapped_segs, colors=colors, linewidths=line_width))

    marker_count = 3
    marker_len = grid_step
    marker_cx = round(pw / 2.0 / grid_step) * grid_step
    marker_cy = round(ph / 2.0 / grid_step) * grid_step
    max_half_h = min(marker_cy, ph - marker_cy) * 0.85
    marker_step = max_half_h / marker_count

    for i in range(1, marker_count + 1):
        half_h = i * marker_step
        half_w = round((half_h * pw / ph) / grid_step) * grid_step
        x1, y1 = marker_cx - half_w, marker_cy - half_h
        x2, y2 = marker_cx + half_w, marker_cy + half_h

        for cx, cy, sx, sy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            h_pts = np.column_stack([
                np.linspace(cx, cx + sx * marker_len, 24),
                np.full(24, cy),
            ])
            v_pts = np.column_stack([
                np.full(24, cx),
                np.linspace(cy, cy + sy * marker_len, 24),
            ])

            for ax, hp, vp in [
                (axes[0, 1], h_pts, v_pts),
                (axes[0, 0], remap_points(h_pts), remap_points(v_pts)),
            ]:
                for color, lw in [("black", 4.5), ("white", 1.5)]:
                    ax.plot(hp[:, 0], hp[:, 1], color=color, linewidth=lw, solid_capstyle="butt", zorder=10)
                    ax.plot(vp[:, 0], vp[:, 1], color=color, linewidth=lw, solid_capstyle="butt", zorder=10)

    axes[0, 0].set_title("Where output grid samples distorted input")
    axes[0, 1].set_title("Regular grid in undistorted output")
    for ax in axes[0]:
        ax.set_xlim(0, pw)
        ax.set_ylim(ph, 0)
        ax.set_aspect("equal")

    axes[1, 0].imshow(cv.cvtColor(preview, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title("Distorted image")
    axes[1, 1].imshow(cv.cvtColor(undistorted, cv.COLOR_BGR2RGB))
    axes[1, 1].set_title("Undistorted image")
    for ax in axes[1]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    out_path = diagnostics_dir(output_dir) / "undistortion_grid.png"
    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_distortion_field(K, D, h, w, camID, output_dir, contour_levels=10, nstep=20):
    """Generate and save a distortion displacement plot for the camera."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    D = D.ravel()
    d = np.zeros(14)
    d[:D.size] = D
    D = d
    k1, k2, p1, p2, k3 = D[0], D[1], D[2], D[3], D[4]

    u, v = np.meshgrid(
        np.arange(0, w, nstep),
        np.arange(0, h, nstep)
    )

    b = np.array([u.ravel(), v.ravel(), np.ones(u.size)])
    xyz = lstsq(K, b, rcond=None)[0]

    xp = xyz[0, :] / xyz[2, :]
    yp = xyz[1, :] / xyz[2, :]
    r2 = xp**2 + yp**2
    r4 = r2**2
    r6 = r2**3

    coef = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + D[5]*r2 + D[6]*r4 + D[7]*r6)
    xpp = xp*coef + 2*p1*(xp*yp) + p2*(r2 + 2*xp**2) + D[8]*r2 + D[9]*r4
    ypp = yp*coef + p1*(r2 + 2*yp**2) + 2*p2*(xp*yp) + D[10]*r2 + D[11]*r4

    u2 = fx*xpp + cx
    v2 = fy*ypp + cy

    du = u2.ravel() - u.ravel()
    dv = v2.ravel() - v.ravel()
    dr = np.hypot(du, dv).reshape(u.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.quiver(u.ravel(), v.ravel(), du, -dv, color="dodgerblue", alpha=0.7)
    ax.plot(w/2, h/2, "x", markersize=10, label="Image center")
    ax.plot(cx, cy, "^", markersize=10, label=f"Principal point ({cx:.1f}, {cy:.1f})")

    CS = ax.contour(u, v, dr, colors="black", levels=contour_levels)
    ax.clabel(CS, inline=1, fontsize=8, fmt="%.0f px")

    ax.set_aspect("equal", "box")
    ax.set_title(f"{camID} Distortion Model\nk1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, p1={p1:.4f}, p2={p2:.4f}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_ylim(max(v.ravel()), 0)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path = diagnostics_dir(output_dir) / "distortion_field.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
