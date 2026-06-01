#!/usr/bin/env python3
"""
stress_test.py — Pruebas de robustez del sistema esteganográfico
=================================================================

Ejemplos:
    python stress_test.py --stego salida.png --key MI_CLAVE --meta salida_meta.json
    python stress_test.py --stego salida.png --key MI_CLAVE --meta salida_meta.json --extended

Notas:
    - El test se considera OK solo si el payload pasa la deserialización y el CRC32.
    - Si falla el CRC, la recuperación no es exacta.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, os.path.dirname(__file__))
import stego_core as sc


def attack_none(img):
    return img.copy()


def attack_grayscale(img):
    return ImageOps.grayscale(img).convert("RGB")


def attack_resize(scale):
    def fn(img):
        w, h = img.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        tmp = img.resize((new_w, new_h), Image.LANCZOS)
        return tmp.resize((w, h), Image.LANCZOS)

    return fn


def attack_jpeg(quality):
    def fn(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    return fn


def attack_webp(quality):
    def fn(img):
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        buf.seek(0)

        return Image.open(buf).convert("RGB")

    return fn


def attack_noise(sigma):
    def fn(img):
        arr = np.array(img.convert("RGB"), dtype=np.float64)
        noise = np.random.normal(0, sigma, arr.shape)

        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    return fn


def attack_blur(radius):
    def fn(img):
        return img.filter(ImageFilter.GaussianBlur(radius=radius)).convert("RGB")

    return fn


def attack_brightness(delta):
    def fn(img):
        arr = np.array(img.convert("RGB"), dtype=np.float64)

        return Image.fromarray(np.clip(arr + delta, 0, 255).astype(np.uint8))

    return fn


def attack_crop_fill(percent):
    """
    Recorta un porcentaje del borde y rellena con negro manteniendo el tamaño original.
    """
    def fn(img):
        w, h = img.size

        dx = int(round(w * percent))
        dy = int(round(h * percent))

        cropped = img.crop((dx, dy, w - dx, h - dy))

        result = Image.new("RGB", (w, h), (0, 0, 0))
        result.paste(cropped, (dx, dy))

        return result

    return fn


def attack_contrast(factor):
    def fn(img):
        return ImageEnhance.Contrast(img).enhance(factor).convert("RGB")

    return fn


def attack_sharpness(factor):
    def fn(img):
        return ImageEnhance.Sharpness(img).enhance(factor).convert("RGB")

    return fn


def attack_gamma(gamma):
    def fn(img):
        arr = np.array(img.convert("RGB"), dtype=np.float64) / 255.0
        arr = np.power(np.clip(arr, 0, 1), gamma)

        return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8))

    return fn


def attack_rotate(degrees):
    def fn(img):
        return img.rotate(
            degrees,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0),
        ).convert("RGB")

    return fn


def build_attacks(extended: bool = False):
    attacks = OrderedDict()

    # Tests básicos similares a los que ya tenías.
    attacks["Original (sin ataque)"] = attack_none
    attacks["Escala de grises → RGB"] = attack_grayscale
    attacks["Reescalado ×0.5 → original"] = attack_resize(0.5)
    attacks["Reescalado ×1.5 → original"] = attack_resize(1.5)
    attacks["Compresión JPEG Q=80"] = attack_jpeg(80)
    attacks["Compresión JPEG Q=50"] = attack_jpeg(50)
    attacks["Ruido gaussiano σ=5"] = attack_noise(5)
    attacks["Desenfoque gaussiano r=2"] = attack_blur(2)
    attacks["Ajuste brillo +30"] = attack_brightness(30)
    attacks["Recorte 5% → relleno"] = attack_crop_fill(0.05)

    if not extended:
        return attacks

    # Tests paramétricos ampliados.
    for q in [95, 90, 80, 70, 60, 50, 40, 30]:
        attacks[f"JPEG Q={q}"] = attack_jpeg(q)

    for q in [90, 80, 70, 60, 50]:
        attacks[f"WebP Q={q}"] = attack_webp(q)

    for sigma in [1, 2, 5, 10, 15, 20]:
        attacks[f"Ruido gaussiano σ={sigma}"] = attack_noise(sigma)

    for radius in [0.5, 1, 1.5, 2, 2.5, 3]:
        attacks[f"Blur gaussiano r={radius}"] = attack_blur(radius)

    for scale in [0.9, 0.75, 0.5, 1.25, 1.5, 2.0]:
        attacks[f"Resize ×{scale} → original"] = attack_resize(scale)

    for percent in [0.01, 0.03, 0.05, 0.10]:
        attacks[f"Crop {int(percent * 100)}% → relleno"] = attack_crop_fill(percent)

    attacks["Contraste ×1.3"] = attack_contrast(1.3)
    attacks["Contraste ×0.7"] = attack_contrast(0.7)
    attacks["Nitidez ×1.5"] = attack_sharpness(1.5)
    attacks["Gamma 1.2"] = attack_gamma(1.2)
    attacks["Gamma 0.8"] = attack_gamma(0.8)
    attacks["Rotación +1°"] = attack_rotate(1)
    attacks["Rotación -1°"] = attack_rotate(-1)

    return attacks


def try_decode(stego_attacked: Image.Image, payload_size: int, key: str, params: dict):
    """
    Devuelve:
        ok, detalle
    """
    try:
        raw = sc.decode(
            stego_attacked,
            payload_size,
            key=key,
            alpha=params["alpha"],
            coeffs=params["coeffs"],
            repeat=params["repeat"],
            safe_margin=params["safe_margin"],
        )

        ptype, data = sc.deserialize_payload(raw)

        ptype_names = {
            sc.PAYLOAD_TEXT: "texto",
            sc.PAYLOAD_AUDIO: "audio",
            sc.PAYLOAD_IMAGE: "imagen",
        }

        return True, f"{len(data):,} bytes ({ptype_names.get(ptype, '?')}) · CRC OK"

    except Exception as e:
        return False, str(e)


def _params_from_profile(profile_name: str):
    profile = sc.PROFILES[profile_name]

    return {
        "alpha": profile["alpha"],
        "coeffs": [tuple(c) for c in profile["coeffs"]],
        "repeat": int(profile["repeat"]),
        "safe_margin": float(profile["safe_margin"]),
    }


def _load_metadata(args):
    payload_size = args.size
    params = _params_from_profile(args.profile)

    meta_path = None

    if args.meta:
        if os.path.exists(args.meta):
            meta_path = args.meta
        else:
            print(f"[WARN] Metadatos no encontrados: {args.meta}. Intentando sin ellos.")

    else:
        root, _ = os.path.splitext(args.stego)
        auto_meta = root + "_meta.json"

        if os.path.exists(auto_meta):
            meta_path = auto_meta

    if meta_path:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        payload_size = meta.get("payload_size_bytes", payload_size)

        params["alpha"] = meta.get("alpha", params["alpha"])
        params["coeffs"] = [tuple(c) for c in meta.get("coeffs", params["coeffs"])]
        params["repeat"] = int(meta.get("repeat", params["repeat"]))
        params["safe_margin"] = float(meta.get("safe_margin", params["safe_margin"]))

    if args.alpha is not None:
        params["alpha"] = args.alpha

    return payload_size, params, meta_path


def main():
    parser = argparse.ArgumentParser(description="Pruebas de robustez esteganográfica")

    parser.add_argument("--stego", required=True, help="Imagen esteganografiada")
    parser.add_argument("--key", default="secreto", help="Clave secreta")
    parser.add_argument("--meta", help="JSON de metadatos")
    parser.add_argument("--size", type=int, help="Tamaño serializado en bytes")
    parser.add_argument(
        "--profile",
        choices=list(sc.PROFILES.keys()),
        default="balanced",
        help="Perfil usado si no hay metadatos",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Sobrescribe alpha de metadatos/perfil",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Ejecutar batería ampliada de ataques",
    )
    parser.add_argument(
        "--save-attacks",
        help="Directorio donde guardar las imágenes atacadas",
    )

    args = parser.parse_args()

    if not os.path.exists(args.stego):
        print(f"[ERROR] No se encuentra: {args.stego}")
        sys.exit(1)

    payload_size, params, meta_path = _load_metadata(args)

    if not payload_size:
        print("[ERROR] Necesito el tamaño serializado del payload. Usa --meta o --size.")
        sys.exit(1)

    stego_img = Image.open(args.stego).convert("RGB")

    if args.save_attacks:
        os.makedirs(args.save_attacks, exist_ok=True)

    attacks = build_attacks(extended=args.extended)

    print(f"\n{'═' * 78}")
    print(f"  PRUEBAS DE ROBUSTEZ — {args.stego}")
    print(f"  Meta={meta_path or 'no'}")
    print(f"  Alpha={params['alpha']}  Repeat={params['repeat']}  Margen={params['safe_margin'] * 100:.1f}%")
    print(f"  Coeffs={params['coeffs']}")
    print(f"  Clave={'*' * len(args.key)}")
    print(f"{'═' * 78}")
    print(f"  {'Ataque':<38} {'Estado':>10}  {'Detalle'}")
    print(f"  {'─' * 38} {'─' * 10}  {'─' * 24}")

    results = OrderedDict()

    for idx, (name, attack_fn) in enumerate(attacks.items(), start=1):
        attacked = attack_fn(stego_img)

        if args.save_attacks:
            safe_name = (
                name.lower()
                .replace(" ", "_")
                .replace("→", "to")
                .replace("×", "x")
                .replace("=", "")
                .replace("%", "pct")
                .replace("°", "deg")
                .replace(".", "_")
                .replace("+", "plus")
                .replace("-", "minus")
            )
            out_path = os.path.join(args.save_attacks, f"{idx:02d}_{safe_name}.png")
            attacked.save(out_path, format="PNG")

        ok, detail = try_decode(attacked, payload_size, args.key, params)

        status = "✅ OK" if ok else "❌ FALLO"

        print(f"  {name:<38} {status:>10}  {detail[:80]}")

        results[name] = ok

    n_ok = sum(results.values())
    n_total = len(results)
    robustness = n_ok / n_total * 100 if n_total else 0

    print(f"{'─' * 78}")
    print(f"  Robustez: {n_ok}/{n_total} pruebas superadas ({robustness:.0f}%)")

    if robustness == 100:
        print("  🏆 Excelente: todos los ataques se han superado con CRC correcto.")
    elif robustness >= 70:
        print("  ✅ Resultado razonable: algunos ataques fuertes degradan el payload.")
    else:
        print("  ⚠️  Robustez insuficiente: prueba profile=robust, más alpha o menos capacidad.")

    print(f"{'═' * 78}\n")


if __name__ == "__main__":
    main()
