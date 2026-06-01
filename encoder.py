#!/usr/bin/env python3
"""
encoder.py — Esteganografía QIM-DCT configurable: CODIFICADOR
=============================================================

Ejemplos:
    python encoder.py --cover imagen.png --payload mensaje.txt --key MI_CLAVE --out salida.png
    python encoder.py --cover imagen.png --payload audio.wav --key MI_CLAVE --out salida.png --profile robust
    python encoder.py --cover imagen.png --payload oculta.png --key MI_CLAVE --out salida.png --profile capacity
    python encoder.py --cover imagen.png --info --profile balanced

Perfiles:
    capacity  -> más capacidad, menos robustez.
    balanced  -> equilibrio.
    robust    -> más robustez, menos capacidad.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import stego_core as sc


def detect_payload_type(filepath: str) -> int:
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".txt", ".md", ".csv", ".json", ".xml", ".html"):
        return sc.PAYLOAD_TEXT

    if ext in (".wav", ".flac", ".ogg", ".aiff"):
        return sc.PAYLOAD_AUDIO

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"):
        return sc.PAYLOAD_IMAGE

    print(f"[!] Extensión '{ext}' no reconocida, tratando como texto.")
    return sc.PAYLOAD_TEXT


def load_payload_bytes(filepath: str, ptype: int) -> bytes:
    if ptype == sc.PAYLOAD_TEXT:
        with open(filepath, "r", encoding="utf-8") as f:
            return sc.text_to_bytes(f.read())

    if ptype == sc.PAYLOAD_AUDIO:
        return sc.audio_to_bytes(filepath)

    if ptype == sc.PAYLOAD_IMAGE:
        img = Image.open(filepath)
        return sc.image_to_bytes(img)

    with open(filepath, "rb") as f:
        return f.read()


def _profile_params(args):
    profile = sc.PROFILES[args.profile]

    alpha = args.alpha if args.alpha is not None else profile["alpha"]
    coeffs = [tuple(c) for c in profile["coeffs"]]
    repeat = int(profile["repeat"])
    safe_margin = float(profile["safe_margin"])

    return alpha, coeffs, repeat, safe_margin


def main():
    parser = argparse.ArgumentParser(
        description="Esteganografía QIM-DCT configurable — Codificador",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--cover", required=True, help="Imagen portadora")
    parser.add_argument("--payload", required=False, help="Archivo a ocultar")
    parser.add_argument("--key", default="secreto", help="Clave secreta")
    parser.add_argument("--out", default="stego_output.png", help="Imagen de salida PNG")

    parser.add_argument(
        "--profile",
        choices=list(sc.PROFILES.keys()),
        default="balanced",
        help="Perfil de codificación",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Sobrescribe el alpha del perfil",
    )

    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="No comprimir el payload antes de ocultarlo",
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Mostrar capacidad sin codificar",
    )

    args = parser.parse_args()

    if not os.path.exists(args.cover):
        print(f"[ERROR] No se encuentra la imagen portadora: {args.cover}")
        sys.exit(1)

    cover_img = Image.open(args.cover)
    w, h = cover_img.size

    alpha, coeffs, repeat, safe_margin = _profile_params(args)

    capacity = sc.get_capacity_bytes(
        cover_img,
        coeffs=coeffs,
        repeat=repeat,
        safe_margin=safe_margin,
    )

    report = sc.get_capacity_report(
        cover_img,
        coeffs=coeffs,
        repeat=repeat,
        safe_margin=safe_margin,
    )

    print(f"\n{'═' * 64}")
    print(f"  ESTEGANOGRAFÍA QIM-DCT — CODIFICADOR v{sc.VERSION}")
    print(f"{'═' * 64}")
    print(f"  Portadora : {args.cover}  ({w}×{h} px)")
    print(f"  Clave     : {'*' * len(args.key)}")
    print(f"  Perfil    : {args.profile}")
    print(f"  Alpha     : {alpha}")
    print(f"  Bloque    : {sc.BLOCK_SIZE}×{sc.BLOCK_SIZE}")
    print(f"  Coeffs    : {coeffs}")
    print(f"  Repeat    : {repeat}")
    print(f"  Margen    : {safe_margin * 100:.1f}%")
    print(f"  Slots     : {report['total_slots']:,}")
    print(f"  Capacidad : {capacity:,} bytes serializados  ({capacity / 1024:.1f} KB)")
    print(f"  Útil aprox: {max(0, capacity - sc.HEADER_SIZE):,} bytes sin cabecera")

    if args.info:
        print(f"{'═' * 64}\n")
        sys.exit(0)

    if not args.payload:
        print("[ERROR] Debes especificar --payload o usar --info.")
        sys.exit(1)

    if not os.path.exists(args.payload):
        print(f"[ERROR] No se encuentra el payload: {args.payload}")
        sys.exit(1)

    ptype = detect_payload_type(args.payload)
    ptype_names = {
        sc.PAYLOAD_TEXT: "Texto",
        sc.PAYLOAD_AUDIO: "Audio",
        sc.PAYLOAD_IMAGE: "Imagen",
    }

    raw_data = load_payload_bytes(args.payload, ptype)

    payload_with_header = sc.serialize_payload(
        ptype,
        raw_data,
        compress=not args.no_compress,
    )

    payload_size = len(payload_with_header)
    compressed = payload_size < len(raw_data) + sc.HEADER_SIZE

    print(f"  Payload   : {args.payload}  (tipo: {ptype_names.get(ptype, 'Desconocido')})")
    print(f"  Original  : {len(raw_data):,} bytes")
    print(f"  Serializ. : {payload_size:,} bytes  ({payload_size / 1024:.1f} KB)")
    print(f"  Compresión: {'sí' if compressed else 'no'}")

    if payload_size > capacity:
        print(f"\n[ERROR] El payload serializado ({payload_size:,} B) supera la capacidad ({capacity:,} B).")
        print("        Usa una imagen más grande, el perfil capacity, o reduce/comprime el payload.")
        sys.exit(1)

    ratio = payload_size / capacity * 100 if capacity else 0
    print(f"  Uso cap.  : {ratio:.1f}%")
    print(f"{'─' * 64}")

    print("  Codificando... ", end="", flush=True)

    try:
        stego_img = sc.encode(
            cover_img,
            payload_with_header,
            key=args.key,
            alpha=alpha,
            coeffs=coeffs,
            repeat=repeat,
            safe_margin=safe_margin,
        )
    except Exception as e:
        print("FALLO")
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("OK")

    out_path = args.out
    stego_img.save(out_path, format="PNG")
    print("  Verificando... ", end="", flush=True)

    try:
        test_img = Image.open(out_path)
        recovered_raw = sc.decode(
            test_img,
            payload_size,
            key=args.key,
            alpha=alpha,
            coeffs=coeffs,
            repeat=repeat,
            safe_margin=safe_margin
            )

        sc.deserialize_payload(recovered_raw)
        print("OK")

    except Exception as e:
        print("FALLO")
        print(f"  [WARN] La imagen se ha generado, pero no se recupera correctamente.")
        print(f"         Motivo: {e}")
        print(f"         Prueba con --alpha más alto o con otro perfil.")
    print(f"  Guardado  : {out_path}")

    orig_arr = np.array(cover_img.convert("RGB"), dtype=np.float64)
    stego_arr = np.array(stego_img.convert("RGB"), dtype=np.float64)

    mse = np.mean((orig_arr - stego_arr) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

    print(f"  PSNR      : {psnr:.2f} dB")

    meta = {
        "payload_size_bytes": payload_size,
        "original_payload_size_bytes": len(raw_data),
        "key_hint": "(proporcionada por el usuario)",
        "block_size": sc.BLOCK_SIZE,
        "coeffs": [list(c) for c in coeffs],
        "repeat": repeat,
        "safe_margin": safe_margin,
        "alpha": alpha,
        "profile": args.profile,
        "header_size": sc.HEADER_SIZE,
        "version": sc.VERSION,
        "compressed_effective": compressed,
    }

    root, _ = os.path.splitext(out_path)
    meta_path = root + "_meta.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Metadatos : {meta_path}  (pásalo al decodificador)")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
