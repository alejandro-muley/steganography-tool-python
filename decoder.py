#!/usr/bin/env python3
"""
decoder.py — Esteganografía QIM-DCT configurable: DESCODIFICADOR
=================================================================

Ejemplos:
    python decoder.py --stego salida.png --key MI_CLAVE --meta salida_meta.json --out recuperado/
    python decoder.py --stego salida.png --key MI_CLAVE --profile robust --out recuperado/
    python decoder.py --stego salida.png --key MI_CLAVE --size 1234 --profile balanced --out recuperado/

Lo recomendable es usar siempre el JSON de metadatos generado por encoder.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from PIL import Image
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
import stego_core as sc


def save_recovered_payload(ptype: int, data: bytes, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)

    if ptype == sc.PAYLOAD_TEXT:
        text = sc.bytes_to_text(data)
        out_path = os.path.join(out_dir, base_name + "_recovered.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  Texto recuperado ({len(text)} caracteres):")
        preview = text[:300] + ("..." if len(text) > 300 else "")

        print(f"  ┌{'─' * 50}┐")
        for line in preview.split("\n")[:8]:
            print(f"  │ {line[:48]:<48} │")
        print(f"  └{'─' * 50}┘")

        return out_path

    elif ptype == sc.PAYLOAD_AUDIO:
        out_path = os.path.join(out_dir, base_name + '_recovered_audio.bin')
        with open(out_path, 'wb') as f:
            f.write(data)

        print(f"  Audio recuperado como bytes: {len(data):,} bytes")
        return out_path

    elif ptype == sc.PAYLOAD_IMAGE:
        img = sc.bytes_to_image(data)
        out_path = os.path.join(out_dir, base_name + "_recovered.png")

        img.save(out_path)
        print(f"  Imagen recuperada: {img.size[0]}×{img.size[1]} px, modo {img.mode}")

        return out_path

    out_path = os.path.join(out_dir, base_name + "_recovered.bin")

    with open(out_path, "wb") as f:
        f.write(data)

    print(f"  Payload binario recuperado: {len(data)} bytes")

    return out_path


def _params_from_profile(profile_name: str):
    profile = sc.PROFILES[profile_name]

    return {
        "alpha": profile["alpha"],
        "coeffs": [tuple(c) for c in profile["coeffs"]],
        "repeat": int(profile["repeat"]),
        "safe_margin": float(profile["safe_margin"]),
    }


def _load_metadata(args):
    """
    Devuelve:
        payload_size, alpha, coeffs, repeat, safe_margin, meta_path_or_none
    """
    payload_size = args.size

    params = _params_from_profile(args.profile)

    meta_path = None

    if args.meta:
        if not os.path.exists(args.meta):
            print(f"[WARN] Metadatos no encontrados: {args.meta}. Intentando sin ellos.")
        else:
            meta_path = args.meta

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

    # Si el usuario escribe --alpha explícitamente, lo dejamos sobrescribir el meta.
    if args.alpha is not None:
        params["alpha"] = args.alpha

    return payload_size, params, meta_path


def main():
    parser = argparse.ArgumentParser(
        description="Esteganografía QIM-DCT configurable — Descodificador",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--stego", required=True, help="Imagen esteganografiada")
    parser.add_argument("--key", default="secreto", help="Clave secreta")
    parser.add_argument("--meta", help="JSON de metadatos del codificador")
    parser.add_argument("--size", type=int, help="Tamaño total serializado en bytes")
    parser.add_argument("--out", default=".", help="Directorio de salida")
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

    args = parser.parse_args()

    print(f"\n{'═' * 64}")
    print(f"  ESTEGANOGRAFÍA QIM-DCT — DESCODIFICADOR v{sc.VERSION}")
    print(f"{'═' * 64}")

    if not os.path.exists(args.stego):
        print(f"[ERROR] No se encuentra: {args.stego}")
        sys.exit(1)

    stego_img = Image.open(args.stego)
    w, h = stego_img.size

    print(f"  Imagen    : {args.stego}  ({w}×{h} px)")
    print(f"  Clave     : {'*' * len(args.key)}")

    payload_size, params, meta_path = _load_metadata(args)

    if meta_path:
        print(f"  Metadatos : {meta_path}")
    else:
        print(f"  Metadatos : no encontrados; usando perfil '{args.profile}'")

    print(f"  Tamaño    : {payload_size:,} bytes" if payload_size else "  Tamaño    : automático")
    print(f"  Alpha     : {params['alpha']}")
    print(f"  Coeffs    : {params['coeffs']}")
    print(f"  Repeat    : {params['repeat']}")
    print(f"  Margen    : {params['safe_margin'] * 100:.1f}%")
    print(f"{'─' * 64}")

    print("  Descodificando... ", end="", flush=True)

    try:
        if payload_size:
            raw = sc.decode(
                stego_img,
                payload_size,
                key=args.key,
                alpha=params["alpha"],
                coeffs=params["coeffs"],
                repeat=params["repeat"],
                safe_margin=params["safe_margin"],
            )

            ptype, data = sc.deserialize_payload(raw)

        else:
            ptype, data = sc.decode_auto(
                stego_img,
                key=args.key,
                alpha=params["alpha"],
                coeffs=params["coeffs"],
                repeat=params["repeat"],
                safe_margin=params["safe_margin"],
            )

        print("OK")

    except Exception as e:
        print("FALLO")
        print(f"[ERROR] {e}")
        sys.exit(1)

    ptype_names = {
        sc.PAYLOAD_TEXT: "Texto",
        sc.PAYLOAD_AUDIO: "Audio",
        sc.PAYLOAD_IMAGE: "Imagen",
    }

    print(f"  Tipo      : {ptype_names.get(ptype, 'Desconocido')}")
    print(f"  Tamaño    : {len(data):,} bytes")
    print(f"  CRC       : OK")

    base = os.path.splitext(os.path.basename(args.stego))[0]
    out_path = save_recovered_payload(ptype, data, args.out, base)

    print(f"  Guardado  : {out_path}")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
