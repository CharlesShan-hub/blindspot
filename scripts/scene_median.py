#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import tqdm
import cv2


def _try_import_toolbox_median():
    try:
        import blindspot as bs  # toolbox
    except Exception:
        return None
    return getattr(bs, "compensate_median", None)


def _fallback_compensate_median(image_2d: np.ndarray, mask_2d: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    result = image_2d.copy()
    rows, cols = image_2d.shape
    pad = kernel_size // 2

    blind_coords = np.argwhere(mask_2d)
    for r, c in blind_coords:
        r_min = max(0, r - pad)
        r_max = min(rows, r + pad + 1)
        c_min = max(0, c - pad)
        c_max = min(cols, c + pad + 1)

        window = image_2d[r_min:r_max, c_min:c_max]
        window_mask = mask_2d[r_min:r_max, c_min:c_max]

        valid_values = window[~window_mask]
        if valid_values.size > 0:
            result[r, c] = np.median(valid_values)

    return result


def _sorted_stems(stems):
    def key_fn(s):
        return (0, int(s)) if s.isdigit() else (1, s)
    return sorted(stems, key=key_fn)


def _collect_by_stem(dir_path: Path, suffix: str, recursive: bool) -> dict[str, Path]:
    if not dir_path.exists():
        raise FileNotFoundError(f"Not found: {dir_path}")
    pattern = f"**/*{suffix}" if recursive else f"*{suffix}"
    paths = [p for p in dir_path.glob(pattern) if p.is_file()]
    return {p.stem: p for p in paths}


def _ensure_uint_like_cast(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = np.rint(arr)
        arr = np.clip(arr, info.min, info.max)
        return arr.astype(dtype)
    return arr.astype(dtype)


def _mask_to_bool(mask_gray: np.ndarray, mode: str, threshold: int) -> np.ndarray:
    if mode not in ("auto", "nonzero_is_blind", "zero_is_blind", "gt"):
        raise ValueError(f"Unknown mask mode: {mode}")

    if mode == "gt":
        return mask_gray > threshold

    nonzero_is_blind = mask_gray != 0
    zero_is_blind = mask_gray == 0

    if mode == "nonzero_is_blind":
        return nonzero_is_blind
    if mode == "zero_is_blind":
        return zero_is_blind

    r1 = float(nonzero_is_blind.mean())
    r2 = float(zero_is_blind.mean())
    return nonzero_is_blind if r1 <= r2 else zero_is_blind


def _save_overlay(img_bgr: np.ndarray, mask_bool: np.ndarray, out_path: Path) -> None:
    overlay = img_bgr.copy()
    red = np.zeros_like(overlay)
    red[..., 2] = 255
    alpha = 0.6
    overlay[mask_bool] = (alpha * red[mask_bool] + (1 - alpha) * overlay[mask_bool]).astype(overlay.dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _inject_blind_pixels(img: np.ndarray, mask_bool: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode not in ("none", "white", "random"):
        raise ValueError(f"Unknown inject mode: {mode}")

    if mode == "none":
        return img

    out = img.copy()
    dtype = out.dtype

    if np.issubdtype(dtype, np.integer):
        maxv = int(np.iinfo(dtype).max)
        minv = int(np.iinfo(dtype).min)
    else:
        maxv = 1.0
        minv = 0.0

    if mode == "white":
        out[mask_bool] = maxv
        return out

    if out.ndim == 2:
        if np.issubdtype(dtype, np.integer):
            out[mask_bool] = rng.integers(low=minv, high=maxv + 1, size=int(mask_bool.sum()), dtype=dtype)
        else:
            out[mask_bool] = rng.uniform(low=minv, high=maxv, size=int(mask_bool.sum())).astype(dtype)
        return out

    if out.ndim == 3:
        ys, xs = np.where(mask_bool)
        n = ys.size
        if n == 0:
            return out
        c = out.shape[2]
        if np.issubdtype(dtype, np.integer):
            vals = rng.integers(low=minv, high=maxv + 1, size=(n, c), dtype=dtype)
        else:
            vals = rng.uniform(low=minv, high=maxv, size=(n, c)).astype(dtype)
        out[ys, xs, :] = vals
        return out

    raise ValueError(f"Unsupported image ndim={out.ndim}")


def main():
    parser = argparse.ArgumentParser(description="Blind-pixel compensation using toolbox median fill (with fallback).")
    parser.add_argument("--indir", required=True, help="Input image directory.")
    parser.add_argument("--maskdir", required=True, help="Mask directory (non-zero = blind pixel).")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--img-suffix", default=".png", help="Image suffix, default: .png")
    parser.add_argument("--mask-suffix", default=".png", help="Mask suffix, default: .png")
    parser.add_argument("--kernel-size", type=int, default=3, help="Median window size, default: 3")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan indir/maskdir.")
    parser.add_argument(
        "--keep-color",
        action="store_true",
        help="Keep 3-channel images as-is. Default is to treat inputs as grayscale (use channel 0) and save grayscale outputs.",
    )
    parser.add_argument(
        "--mask-mode",
        default="auto",
        choices=["auto", "nonzero_is_blind", "zero_is_blind", "gt"],
        help="How to interpret mask: auto picks the sparser one; gt uses --mask-threshold.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="Used when --mask-mode gt (mask > thr is blind).")
    parser.add_argument("--resize-mask", action="store_true", help="If mask size != image size, resize mask with nearest.")
    parser.add_argument("--out-ext", default=None, help="Output extension (e.g. .png). Default: keep input extension.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--inject-mode",
        default="random",
        choices=["none", "white", "random"],
        help="Before compensation, overwrite blind pixels in the input image to simulate corruption.",
    )
    parser.add_argument(
        "--inject-seed",
        type=int,
        default=0,
        help="Seed for random injection. Use -1 for non-deterministic.",
    )
    parser.add_argument("--save-corrupted", action="store_true", help="Also save corrupted images to outdir/_corrupted.")
    parser.add_argument("--debug-limit", type=int, default=5, help="Print mask stats for first N files.")
    parser.add_argument("--debug-overlay", action="store_true", help="Save overlay images with blind pixels in red.")
    args = parser.parse_args()

    if args.kernel_size < 1 or args.kernel_size % 2 == 0:
        raise ValueError("--kernel-size must be an odd integer >= 1")

    indir = Path(args.indir)
    maskdir = Path(args.maskdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = _collect_by_stem(indir, args.img_suffix, args.recursive)
    masks = _collect_by_stem(maskdir, args.mask_suffix, args.recursive)

    common = set(imgs.keys()) & set(masks.keys())
    if not common:
        raise RuntimeError(f"No common stems between {indir} ({args.img_suffix}) and {maskdir} ({args.mask_suffix})")

    stems = _sorted_stems(common)

    toolbox_fn = _try_import_toolbox_median()
    compensate_median = toolbox_fn if toolbox_fn is not None else _fallback_compensate_median

    rng = np.random.default_rng(None if args.inject_seed < 0 else args.inject_seed)

    for i, stem in enumerate(tqdm.tqdm(stems, desc="median-fill")):
        img_path = imgs[stem]
        mask_path = masks[stem]

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        if img.ndim == 3 and not args.keep_color:
            img = img[..., 0]

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if mask_gray.shape[:2] != img.shape[:2]:
            if not args.resize_mask:
                raise RuntimeError(
                    f"Mask size {mask_gray.shape[:2]} != image size {img.shape[:2]} for stem={stem}. "
                    f"Use --resize-mask if this is expected."
                )
            mask_gray = cv2.resize(mask_gray, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask_bool = _mask_to_bool(mask_gray, args.mask_mode, args.mask_threshold)

        if i < args.debug_limit:
            uniq = np.unique(mask_gray)
            blind_count = int(mask_bool.sum())
            total = int(mask_bool.size)
            print(
                f"[{stem}] mask unique={uniq[:20]}{'...' if uniq.size > 20 else ''} "
                f"blind={blind_count}/{total} ({blind_count/total:.6f}) mode={args.mask_mode}"
            )

        out_ext = args.out_ext if args.out_ext is not None else img_path.suffix
        out_path = outdir / f"{stem}{out_ext}"
        if out_path.exists() and not args.overwrite:
            continue

        img_corrupt = _inject_blind_pixels(img, mask_bool, args.inject_mode, rng)

        if args.save_corrupted:
            corrupt_path = outdir / "_corrupted" / f"{stem}{out_ext}"
            corrupt_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(corrupt_path), img_corrupt)
            if not ok:
                raise RuntimeError(f"Failed to write corrupted image: {corrupt_path}")

        if img_corrupt.ndim == 2:
            dtype = img_corrupt.dtype
            comp = compensate_median(img_corrupt.astype(np.float32), mask_bool.astype(bool), kernel_size=args.kernel_size)
            comp = _ensure_uint_like_cast(comp, dtype)
        else:
            dtype = img_corrupt.dtype
            comp = np.empty_like(img_corrupt, dtype=dtype)
            for ch in range(img_corrupt.shape[2]):
                ch_comp = compensate_median(
                    img_corrupt[..., ch].astype(np.float32),
                    mask_bool.astype(bool),
                    kernel_size=args.kernel_size,
                )
                comp[..., ch] = _ensure_uint_like_cast(ch_comp, dtype)

        ok = cv2.imwrite(str(out_path), comp)
        if not ok:
            raise RuntimeError(f"Failed to write output: {out_path}")

        if args.debug_overlay:
            img_bgr = (
                img_corrupt
                if (img_corrupt.ndim == 3 and img_corrupt.shape[2] == 3)
                else cv2.cvtColor(img_corrupt, cv2.COLOR_GRAY2BGR)
            )
            overlay_path = outdir / "_debug_overlay" / f"{stem}.png"
            _save_overlay(img_bgr, mask_bool, overlay_path)


if __name__ == "__main__":
    main()
