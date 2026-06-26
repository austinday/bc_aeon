#!/usr/bin/env python3
"""
Normalize an AtomicChat 'gemma-4-31B-it-assistant' MTP GGUF so it loads with the
atomic-llama-cpp-turboquant fork's `gemma4-assistant` loader.

The published HF file (AtomicChat/gemma-4-31B-it-assistant-GGUF) is the correct
*model* but uses a different naming convention than the fork's C++ loader expects:

  metadata:
    general.architecture            : gemma4_assistant   -> gemma4-assistant   (+ same for every `<arch>.*` key prefix)
    (missing) <arch>.nextn_predict_layers                -> = block_count      (all-nextn standalone draft)
    (missing) <arch>.embedding_length_out                -> = <arch>.n_embd_backbone (target hidden size)
  tensors:
    mtp.pre_projection.weight       -> nextn.pre_projection.weight
    mtp.post_projection.weight      -> nextn.post_projection.weight

Tensor *data* is copied verbatim (dims/types/order unchanged), so this is a pure
metadata/name relabel. Idempotent: if the file already looks normalized it is copied
through unchanged. Pure stdlib (no gguf/numpy dependency).
"""
import struct, sys, os

# GGUF metadata value type ids
U8,I8,U16,I16,U32,I32,F32,BOOL,STRING,ARRAY,U64,I64,F64 = range(13)
_FIXED = {U8:1,I8:1,U16:2,I16:2,U32:4,I32:4,F32:4,BOOL:1,U64:8,I64:8,F64:8}


def normalize(src_path, dst_path):
    b = open(src_path, "rb").read()
    p = 0
    def take(n):
        nonlocal p
        s = b[p:p+n]; p += n; return s
    def ru32(): return struct.unpack("<I", take(4))[0]
    def ru64(): return struct.unpack("<Q", take(8))[0]
    def rstr():
        l = ru64(); return take(l)
    def skip_value(t):
        nonlocal p
        if t in _FIXED: p += _FIXED[t]
        elif t == STRING: rstr()
        elif t == ARRAY:
            et = ru32(); cnt = ru64()
            for _ in range(cnt): skip_value(et)
        else: raise ValueError("bad type %d" % t)

    assert take(4) == b"GGUF", "not a GGUF file"
    ver = ru32()
    assert ver == 3, "only GGUF v3 supported (got %d)" % ver
    n_tensors = ru64()
    n_kv = ru64()

    align = 32
    kv_start = p
    kv = []  # (key bytes, raw entry bytes)
    arch = None
    block_count = None
    n_embd_backbone = None
    for _ in range(n_kv):
        e0 = p
        key = rstr()
        t = ru32()
        v0 = p
        skip_value(t)
        entry = b[e0:p]
        kv.append([key, entry])
        if key == b"general.alignment" and t == U32:
            align = struct.unpack("<I", b[v0:v0+4])[0]

    # decode a couple of values we need (after we know the arch)
    def decode_scalar(entry):
        # entry = <u64 keylen><key><u32 type><value...>
        kl = struct.unpack("<Q", entry[:8])[0]
        t = struct.unpack("<I", entry[8+kl:8+kl+4])[0]
        vp = 8+kl+4
        if t in (U32, I32): return struct.unpack("<i", entry[vp:vp+4])[0]
        if t in (U64, I64): return struct.unpack("<q", entry[vp:vp+8])[0]
        return None

    # find arch
    for key, entry in kv:
        if key == b"general.architecture":
            kl = struct.unpack("<Q", entry[:8])[0]
            # value is a string
            vp = 8+kl+4
            sl = struct.unpack("<Q", entry[vp:vp+8])[0]
            arch = entry[vp+8:vp+8+sl]
            break
    assert arch is not None, "no general.architecture"

    norm_arch = arch.replace(b"_", b"-")  # gemma4_assistant -> gemma4-assistant
    # relabel arch in all kv keys/values (same length so offsets within entry unchanged)
    new_kv = []
    have_nextn = False
    have_embd_out = False
    for key, entry in kv:
        nkey = key.replace(arch, norm_arch)
        nentry = entry.replace(arch, norm_arch)  # also fixes general.architecture value
        new_kv.append([nkey, nentry])
        if nkey == norm_arch + b".block_count":
            block_count = decode_scalar(nentry)
        if nkey == norm_arch + b".n_embd_backbone":
            n_embd_backbone = decode_scalar(nentry)
        if nkey == norm_arch + b".nextn_predict_layers":
            have_nextn = True
        if nkey == norm_arch + b".embedding_length_out":
            have_embd_out = True

    def make_u32_kv(key, val):
        return struct.pack("<Q", len(key)) + key + struct.pack("<I", U32) + struct.pack("<I", val)

    added = 0
    if not have_nextn:
        assert block_count is not None, "block_count missing; cannot set nextn_predict_layers"
        new_kv.append([norm_arch + b".nextn_predict_layers",
                       make_u32_kv(norm_arch + b".nextn_predict_layers", block_count)])
        added += 1
    if not have_embd_out:
        assert n_embd_backbone is not None, "n_embd_backbone missing; cannot set embedding_length_out"
        new_kv.append([norm_arch + b".embedding_length_out",
                       make_u32_kv(norm_arch + b".embedding_length_out", n_embd_backbone)])
        added += 1

    # ---- tensor infos ----
    tinfo_start = p
    tensors = []  # [name, raw_dims_bytes, type, offset]
    for _ in range(n_tensors):
        name = rstr()
        nd = ru32()
        dims = take(8*nd)
        ttype = ru32()
        toff = ru64()
        tensors.append([name, nd, dims, ttype, toff])
    tinfo_end = p
    data_off = (tinfo_end + align - 1) // align * align
    data = b[data_off:]

    RENAME = {
        b"mtp.pre_projection.weight":  b"nextn.pre_projection.weight",
        b"mtp.post_projection.weight": b"nextn.post_projection.weight",
    }
    renamed = 0
    for t in tensors:
        if t[0] in RENAME:
            t[0] = RENAME[t[0]]; renamed += 1

    # ---- emit ----
    out = bytearray()
    out += b"GGUF" + struct.pack("<I", ver)
    out += struct.pack("<Q", n_tensors)
    out += struct.pack("<Q", len(new_kv))
    for _key, entry in new_kv:
        out += entry
    for name, nd, dims, ttype, toff in tensors:
        out += struct.pack("<Q", len(name)) + name
        out += struct.pack("<I", nd) + dims
        out += struct.pack("<I", ttype) + struct.pack("<Q", toff)
    pad = (-len(out)) % align
    out += b"\x00" * pad
    out += data

    with open(dst_path, "wb") as f:
        f.write(out)
    print("[normalize] arch %s -> %s | kv +%d | tensors renamed %d | %d -> %d bytes"
          % (arch.decode(), norm_arch.decode(), added, renamed, len(b), len(out)))
    print("[normalize] block_count=%s n_embd_backbone=%s" % (block_count, n_embd_backbone))
    return added, renamed


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: normalize_gemma4_assistant.py <src.gguf> <dst.gguf>", file=sys.stderr)
        sys.exit(2)
    normalize(sys.argv[1], sys.argv[2])
