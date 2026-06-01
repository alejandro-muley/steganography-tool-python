"""
stego_core.py — Motor de esteganografía QIM-DCT configurable
=============================================================

Versión mejorada respecto al motor original:

- Varios coeficientes DCT por bloque.
- Perfiles de capacidad/robustez.
- Repetición de bits con voto por mayoría.
- Margen seguro para evitar bordes y resistir recortes.
- Compresión opcional con zlib.
- CRC32 para verificar si el payload recuperado es exactamente correcto.

IMPORTANTE:
Este motor sigue siendo QIM-DCT. No usa DWT. Si se quiere DWT+DCT real,
habría que añadir una transformada wavelet, por ejemplo con PyWavelets.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from scipy.fft import dct, idct


# ─────────────────────────────────────────────────
#  PARÁMETROS GLOBALES
# ─────────────────────────────────────────────────

ALPHA = 60.0
BLOCK_SIZE = 8

# Coeficientes DCT por defecto. Se evita (0,0), que es la componente DC.
COEFFS = [(1, 2), (2, 1), (2, 2), (1, 3)]

# Repeticiones por defecto para voto por mayoría.
REPEAT = 3

# Fracción de borde que se evita. 0.05 = no usar el 5% exterior.
SAFE_MARGIN = 0.05

MAGIC = b"STEG"
VERSION = 3

# Cabecera:
# [MAGIC 4B][VERSION 1B][TYPE 1B][FLAGS 1B][LENGTH 4B][CRC32 4B]
HEADER_SIZE = 15

FLAG_COMPRESSED = 0x01

# Tipos de payload.
PAYLOAD_TEXT = 0x01
PAYLOAD_AUDIO = 0x02
PAYLOAD_IMAGE = 0x03


PROFILES = {
    # Máxima capacidad: muchos coeficientes, sin repetición, sin margen.
    # Menos robusto contra ataques fuertes.
    "capacity": {
        "alpha": 50.0,
        "coeffs": [
        (2, 1), (1, 2), (2, 2),
        (1, 3), (3, 1), (2, 3)],
        "repeat": 2,
        "safe_margin": 0.00,
    },

    # Equilibrio razonable entre capacidad y robustez.
    "balanced": {
        "alpha": 60.0,
        "coeffs": [(1, 2), (2, 1), (2, 2), (1, 3)],
        "repeat": 3,
        "safe_margin": 0.05,
    },

    # Más robusto: menos coeficientes, más repetición y margen mayor.
    # Menor capacidad efectiva.
    "robust": {
        "alpha": 90.0,
        "coeffs": [(1, 2), (2, 1), (2, 2)],
        "repeat": 7,
        "safe_margin": 0.10,
    },
}


# ─────────────────────────────────────────────────
#  VALIDACIÓN / NORMALIZACIÓN
# ─────────────────────────────────────────────────

def _normalize_coeffs(coeffs: Sequence[Sequence[int]] | None) -> list[tuple[int, int]]:
    if coeffs is None:
        coeffs = COEFFS

    normalized = []
    for c in coeffs:
        if len(c) != 2:
            raise ValueError(f"Coeficiente DCT inválido: {c!r}")

        r, col = int(c[0]), int(c[1])

        if not (0 <= r < BLOCK_SIZE and 0 <= col < BLOCK_SIZE):
            raise ValueError(f"Coeficiente fuera del bloque {BLOCK_SIZE}x{BLOCK_SIZE}: {(r, col)}")

        if (r, col) == (0, 0):
            raise ValueError("No se recomienda usar el coeficiente DC (0,0).")

        normalized.append((r, col))

    if not normalized:
        raise ValueError("La lista de coeficientes no puede estar vacía.")

    return normalized


def _validate_repeat(repeat: int) -> int:
    repeat = int(repeat)

    if repeat < 1:
        raise ValueError("repeat debe ser >= 1.")

    return repeat


def _validate_safe_margin(safe_margin: float) -> float:
    safe_margin = float(safe_margin)

    if not (0.0 <= safe_margin < 0.5):
        raise ValueError("safe_margin debe estar en [0.0, 0.5).")

    return safe_margin


# ─────────────────────────────────────────────────
#  UTILIDADES CRIPTOGRÁFICAS
# ─────────────────────────────────────────────────

def _key_to_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)


def _generate_permutation(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(n)


# ─────────────────────────────────────────────────
#  SERIALIZACIÓN
# ─────────────────────────────────────────────────

def serialize_payload(payload_type: int, data: bytes, compress: bool = True) -> bytes:
    """
    Serializa el payload.

    Formato:
        [MAGIC 4B][VERSION 1B][TYPE 1B][FLAGS 1B][LENGTH 4B][CRC32 4B][DATA...]

    LENGTH es el tamaño de DATA guardado, que puede estar comprimido.
    CRC32 se calcula sobre los datos originales sin comprimir.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data debe ser bytes.")

    payload_type = int(payload_type)
    flags = 0

    original_data = bytes(data)
    original_crc = zlib.crc32(original_data) & 0xFFFFFFFF

    stored_data = original_data

    if compress:
        compressed = zlib.compress(original_data, level=9)

        # Solo guardamos comprimido si realmente ahorra espacio.
        if len(compressed) < len(original_data):
            stored_data = compressed
            flags |= FLAG_COMPRESSED

    header = MAGIC + struct.pack(
        ">BBBII",
        VERSION,
        payload_type,
        flags,
        len(stored_data),
        original_crc,
    )

    return header + stored_data


def _parse_header(raw: bytes):
    if len(raw) < HEADER_SIZE:
        raise ValueError("Datos demasiado cortos para contener la cabecera.")

    if raw[:4] != MAGIC:
        raise ValueError(
            f"Cabecera inválida: {raw[:4]!r}. "
            "¿Clave incorrecta, parámetros incorrectos o imagen no esteganografiada?"
        )

    version, ptype, flags, length, expected_crc = struct.unpack(
        ">BBBII",
        raw[4:HEADER_SIZE],
    )

    if version != VERSION:
        raise ValueError(
            f"Versión incompatible: encontrada {version}, esperada {VERSION}."
        )

    if length < 0:
        raise ValueError(f"Tamaño de payload inválido: {length} bytes.")

    return version, ptype, flags, length, expected_crc


def deserialize_payload(raw: bytes):
    """
    Deserializa el payload y valida CRC32.

    Devuelve:
        (payload_type, data_original)
    """
    _, ptype, flags, length, expected_crc = _parse_header(raw)

    end = HEADER_SIZE + length
    if len(raw) < end:
        raise ValueError(
            f"Payload incompleto: se esperaban {end} bytes serializados "
            f"y solo hay {len(raw)}."
        )

    stored_data = raw[HEADER_SIZE:end]

    if flags & FLAG_COMPRESSED:
        try:
            data = zlib.decompress(stored_data)
        except zlib.error as e:
            raise ValueError(f"No se pudo descomprimir el payload: {e}") from e
    else:
        data = stored_data

    actual_crc = zlib.crc32(data) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            f"CRC inválido. Payload corrupto. "
            f"Esperado={expected_crc}, obtenido={actual_crc}."
        )

    return ptype, data


# ─────────────────────────────────────────────────
#  CONVERSIÓN DE PAYLOADS
# ─────────────────────────────────────────────────

def text_to_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def bytes_to_text(data: bytes) -> str:
    return data.decode("utf-8")


def image_to_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def bytes_to_image(data: bytes) -> Image.Image:
    import io

    return Image.open(io.BytesIO(data))


def audio_to_bytes(filepath: str) -> bytes:
    with open(filepath, "rb") as f:
        return f.read()


def bytes_to_audio(data: bytes):
    import io
    import soundfile as sf

    if len(data) < 4:
        raise ValueError("Datos de audio demasiado cortos.")

    samplerate = struct.unpack(">i", data[:4])[0]
    samples, _ = sf.read(io.BytesIO(data[4:]), dtype="float32")

    return samples, samplerate


# ─────────────────────────────────────────────────
#  BITS
# ─────────────────────────────────────────────────

def _bytes_to_symbols(data: bytes) -> np.ndarray:
    """
    Convierte bytes a símbolos:
        bit 0 -> -1.0
        bit 1 -> +1.0
    """
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.float64) * 2.0 - 1.0


def _symbols_to_bytes(symbols: np.ndarray) -> bytes:
    """
    Convierte símbolos:
        <= 0 -> bit 0
        > 0  -> bit 1
    """
    bits = (symbols > 0).astype(np.uint8)
    pad = (8 - len(bits) % 8) % 8

    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


# ─────────────────────────────────────────────────
#  QIM-DCT POR BLOQUE
# ─────────────────────────────────────────────────

def _dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: np.ndarray) -> np.ndarray:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def _embed_bit(
    block: np.ndarray,
    bit_val: float,
    alpha: float,
    coeff_idx: tuple[int, int],
) -> np.ndarray:
    """
    QIM sobre un coeficiente DCT.

    bit +1 -> índice cuantizado impar
    bit -1 -> índice cuantizado par
    """
    B = _dct2(block)

    qi = int(round(B[coeff_idx] / alpha))
    target = 1 if bit_val > 0 else 0

    if abs(qi) % 2 != target:
        qi = qi + 1 if qi >= 0 else qi - 1

    B[coeff_idx] = qi * alpha

    return _idct2(B)


def _extract_bit(
    block: np.ndarray,
    alpha: float,
    coeff_idx: tuple[int, int],
) -> float:
    B = _dct2(block)

    qi = int(round(B[coeff_idx] / alpha))

    return 1.0 if abs(qi) % 2 == 1 else -1.0


# ─────────────────────────────────────────────────
#  BLOQUES, SLOTS Y CAPACIDAD
# ─────────────────────────────────────────────────

def _safe_block_indices(h: int, w: int, safe_margin: float = 0.0) -> np.ndarray:
    """
    Devuelve los índices de bloques que se pueden usar.

    Si safe_margin > 0, evita los bordes. Esto ayuda contra recortes.
    """
    safe_margin = _validate_safe_margin(safe_margin)

    bh = h // BLOCK_SIZE
    bw = w // BLOCK_SIZE

    margin_y = int(math.ceil(bh * safe_margin))
    margin_x = int(math.ceil(bw * safe_margin))

    if margin_y * 2 >= bh or margin_x * 2 >= bw:
        return np.array([], dtype=np.int64)

    indices = []

    for by in range(margin_y, bh - margin_y):
        for bx in range(margin_x, bw - margin_x):
            indices.append(by * bw + bx)

    return np.array(indices, dtype=np.int64)


def _slot_positions(
    h: int,
    w: int,
    n_bits: int,
    seed: int,
    coeffs: Sequence[Sequence[int]] | None,
    safe_margin: float,
):
    """
    Un slot es:
        (bloque 8x8, coeficiente DCT)

    Si hay 4 coeficientes por bloque, cada bloque tiene 4 slots.
    """
    coeffs = _normalize_coeffs(coeffs)
    blocks = _safe_block_indices(h, w, safe_margin)

    n_blocks = len(blocks)
    n_coeffs = len(coeffs)
    n_slots = n_blocks * n_coeffs

    if n_bits > n_slots:
        raise ValueError(
            f"Payload demasiado grande: {n_bits} bits, disponibles {n_slots} slots."
        )

    perm = _generate_permutation(n_slots, seed)[:n_bits]

    block_ids = blocks[perm // n_coeffs]
    coeff_ids = perm % n_coeffs

    return block_ids, coeff_ids


def get_capacity_bytes(
    img: Image.Image,
    coeffs: Sequence[Sequence[int]] | None = None,
    repeat: int = 1,
    safe_margin: float = 0.0,
) -> int:
    """
    Capacidad máxima en bytes serializados.

    Incluye cabecera. Es decir, si devuelve 2000, el resultado de
    serialize_payload(...) debe ocupar como máximo 2000 bytes.
    """
    coeffs = _normalize_coeffs(coeffs)
    repeat = _validate_repeat(repeat)

    w, h = img.size
    blocks = _safe_block_indices(h, w, safe_margin)

    n_slots = len(blocks) * len(coeffs)
    usable_bits = n_slots // repeat
    usable_bytes = usable_bits // 8

    return max(0, usable_bytes)


def get_capacity_report(
    img: Image.Image,
    coeffs: Sequence[Sequence[int]] | None = None,
    repeat: int = 1,
    safe_margin: float = 0.0,
) -> dict:
    coeffs = _normalize_coeffs(coeffs)
    repeat = _validate_repeat(repeat)

    w, h = img.size
    blocks = _safe_block_indices(h, w, safe_margin)

    total_serialized = get_capacity_bytes(
        img,
        coeffs=coeffs,
        repeat=repeat,
        safe_margin=safe_margin,
    )

    useful_approx = max(0, total_serialized - HEADER_SIZE)

    return {
        "width": w,
        "height": h,
        "block_size": BLOCK_SIZE,
        "usable_blocks": int(len(blocks)),
        "coeffs_per_block": int(len(coeffs)),
        "repeat": int(repeat),
        "safe_margin": float(safe_margin),
        "total_slots": int(len(blocks) * len(coeffs)),
        "capacity_serialized_bytes": int(total_serialized),
        "capacity_payload_approx_bytes": int(useful_approx),
    }


# ─────────────────────────────────────────────────
#  API PÚBLICA
# ─────────────────────────────────────────────────

def encode(
    cover_img: Image.Image,
    payload_bytes: bytes,
    key: str = "secreto",
    alpha: float | None = None,
    coeffs: Sequence[Sequence[int]] | None = None,
    repeat: int | None = None,
    safe_margin: float | None = None,
) -> Image.Image:
    """
    Oculta payload_bytes en cover_img.

    Trabaja sobre el canal Y de YCbCr.
    """
    if alpha is None:
        alpha = ALPHA

    if repeat is None:
        repeat = REPEAT

    if safe_margin is None:
        safe_margin = SAFE_MARGIN

    alpha = float(alpha)
    coeffs = _normalize_coeffs(coeffs)
    repeat = _validate_repeat(repeat)
    safe_margin = _validate_safe_margin(safe_margin)

    ycbcr = np.array(cover_img.convert("RGB").convert("YCbCr"), dtype=np.float64)
    Y = ycbcr[:, :, 0]

    h, w = Y.shape
    bw_count = w // BLOCK_SIZE

    symbols = _bytes_to_symbols(payload_bytes)

    if repeat > 1:
        symbols = np.repeat(symbols, repeat)

    block_ids, coeff_ids = _slot_positions(
        h,
        w,
        len(symbols),
        _key_to_seed(key),
        coeffs,
        safe_margin,
    )

    Y_out = Y.copy()

    for i in range(len(symbols)):
        block_id = int(block_ids[i])
        coeff_idx = coeffs[int(coeff_ids[i])]

        r0 = (block_id // bw_count) * BLOCK_SIZE
        c0 = (block_id % bw_count) * BLOCK_SIZE

        block = Y_out[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE]

        Y_out[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE] = _embed_bit(
            block,
            symbols[i],
            alpha,
            coeff_idx,
        )

    ycbcr[:, :, 0] = np.clip(Y_out, 0, 255)

    return Image.fromarray(ycbcr.astype(np.uint8), "YCbCr").convert("RGB")


def decode(
    stego_img: Image.Image,
    n_payload_bytes: int,
    key: str = "secreto",
    alpha: float | None = None,
    coeffs: Sequence[Sequence[int]] | None = None,
    repeat: int | None = None,
    safe_margin: float | None = None,
) -> bytes:
    """
    Extrae n_payload_bytes bytes serializados de la imagen esteganografiada.
    """
    if alpha is None:
        alpha = ALPHA

    if repeat is None:
        repeat = REPEAT

    if safe_margin is None:
        safe_margin = SAFE_MARGIN

    alpha = float(alpha)
    coeffs = _normalize_coeffs(coeffs)
    repeat = _validate_repeat(repeat)
    safe_margin = _validate_safe_margin(safe_margin)

    Y = np.array(stego_img.convert("RGB").convert("YCbCr"), dtype=np.float64)[:, :, 0]

    h, w = Y.shape
    bw_count = w // BLOCK_SIZE

    original_n_bits = int(n_payload_bytes) * 8
    n_bits_to_read = original_n_bits * repeat

    block_ids, coeff_ids = _slot_positions(
        h,
        w,
        n_bits_to_read,
        _key_to_seed(key),
        coeffs,
        safe_margin,
    )

    recovered = np.zeros(n_bits_to_read, dtype=np.float64)

    for i in range(n_bits_to_read):
        block_id = int(block_ids[i])
        coeff_idx = coeffs[int(coeff_ids[i])]

        r0 = (block_id // bw_count) * BLOCK_SIZE
        c0 = (block_id % bw_count) * BLOCK_SIZE

        block = Y[r0:r0 + BLOCK_SIZE, c0:c0 + BLOCK_SIZE]

        recovered[i] = _extract_bit(block, alpha, coeff_idx)

    if repeat > 1:
        recovered = recovered.reshape(original_n_bits, repeat)
        recovered = np.where(np.sum(recovered, axis=1) >= 0, 1.0, -1.0)

    return _symbols_to_bytes(recovered)[:n_payload_bytes]


def decode_auto(
    stego_img: Image.Image,
    key: str = "secreto",
    alpha: float | None = None,
    coeffs: Sequence[Sequence[int]] | None = None,
    repeat: int | None = None,
    safe_margin: float | None = None,
    max_payload_kb: int = 4096,
):
    """
    Decodificación automática.

    Primero lee la cabecera, obtiene el tamaño de DATA y luego lee el payload completo.
    Devuelve:
        (payload_type, data_bytes)
    """
    if alpha is None:
        alpha = ALPHA

    if repeat is None:
        repeat = REPEAT

    if safe_margin is None:
        safe_margin = SAFE_MARGIN

    raw_header = decode(
        stego_img,
        HEADER_SIZE,
        key=key,
        alpha=alpha,
        coeffs=coeffs,
        repeat=repeat,
        safe_margin=safe_margin,
    )

    _, _, _, length, _ = _parse_header(raw_header)

    if length <= 0 or length > max_payload_kb * 1024:
        raise ValueError(f"Tamaño de payload inválido: {length} bytes.")

    raw_full = decode(
        stego_img,
        HEADER_SIZE + length,
        key=key,
        alpha=alpha,
        coeffs=coeffs,
        repeat=repeat,
        safe_margin=safe_margin,
    )

    return deserialize_payload(raw_full)
