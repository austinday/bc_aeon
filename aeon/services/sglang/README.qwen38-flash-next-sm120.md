# Qwen3.8-Flash-Next SM120 SGLang runtime overlay

This recipe prepares the derived runtime image for one RTX PRO 6000 Blackwell
96 GB card. It does **not** qualify or promote that image. The source inputs and
local overlays below are pinned; GPU startup, semantic, memory, and throughput
qualification remain separate gates.

## Source closure

- Official model source and architecture notes:
  [QwenLM/Qwen3.8-Flash-Next](https://github.com/QwenLM/Qwen3.8-Flash-Next)
- Official SGLang day-zero cookbook:
  [Qwen3.8-Flash-Next](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- NVIDIA's unified-Hugging-Face ModelOpt/SGLang contract:
  [Deploy Quantized Models](https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html)
- Public day-zero image index:
  `lmsysorg/sglang:qwen38flashnext@sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1`
- Exact linux/amd64 child used by the Dockerfile:
  `sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae`
- Image-declared base SGLang commit:
  `d91c3682b0b429e4c70df63cd57f819588ce29b0`
- Image-declared Qwen overlay: `Qiaolin-Yu/sglang-qwen-next#38`, commits
  `3ea3a37a1,12070370f`
- Public qwen4_exp reference implementation:
  [SGLang PR 36497](https://github.com/sgl-project/sglang/pull/36497), commit
  `73a255206f916366c8d26d4022f82ddfb0ab558d`
- Required RTX PRO 6000/SM120 QSA fix:
  [SGLang PR 36556](https://github.com/sgl-project/sglang/pull/36556), commit
  `dac5523d1e5d2f4297fec40ef02fc76fb0f662d1`
- Applied patch SHA-256:
  `eba9b1b2c07f6bdfe42502ffc50667f7e1e5467dc1ee96f0a8e791562e1c9679`
- Official SM120 ModelOpt-FP4 backend selection:
  [SGLang PR 30075](https://github.com/sgl-project/sglang/pull/30075), commit
  `3836cba9eed2cc0db093e58ca839215609a44c31`
- Open SGLang PR 36601 Qwen3.8 fused shared-expert loader change: commit
  `cdb7ac8f4740f0baf5d01d673fd0fb671a14ebdf`, applied Qwen4 patch SHA-256
  `9c3d91412bd3599ccfb5a8879448423fbc34cc24659593933dabe22858ce7338`
- Open SGLang PR 36601 Qwen3.8 MTP shared-expert loader change: commit
  `7db597910dab20741770862d328c1399be0e6ab8`, applied day-zero adaptation
  SHA-256
  `e9f26827b1c0da319c1116caea575b89a794c983ed35671331d421d40137b7fb`
- Local CUTLASS scale-placeholder headroom overlay SHA-256:
  `a6c61ef9eaa1153551506b26aca7627f7ecc98851f6cd7e7038cd6d0a25b5c6a`
- Official-repository SGLang issue proposal
  [#36452](https://github.com/sgl-project/sglang/issues/36452), moving MTP
  embed/head sharing before memory-pool sizing; local overlay SHA-256:
  `424eb761834646089437f7e2d16694ab06f03e102f045da07f4a35aa3c83b607`
- Canonical composed source-stack SHA-256:
  `f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed`

The public day-zero image alone is rejected on SM120. PR 36556 routes Qwen
sparse decode through the SM120-capable TRT-LLM path and uses SGLang's owned FA4
dispatcher for its packed fallback. The pinned day-zero image omits SGLang's
`test/registered` tree, so the Dockerfile hashes the complete upstream patch but
applies only its exact runtime-file diff with `git apply --include`; the upstream
test hunk remains review evidence and the local CPU contract suite covers the
same dispatch invariants. The Qwen4 shared-expert patch is likewise narrowed to
its runtime file. The MTP patch is an explicit day-zero adaptation of the two
upstream loader commits because that image predates their current-main layout.
The build hashes every patch and fails if a runtime diff no longer applies
exactly. The headroom overlay does not delete a live scale: current SGLang's
post-load `alias_or_bind_derived_param` already stores the CUTLASS swizzle in the
checkpoint scale buffer for this no-padding geometry. It prevents the earlier
eager allocation of 7,549,747,200 bytes of uninitialized swizzled placeholders,
whose released blocks otherwise remain as fragmented CUDA allocator reserve.
The MTP timing overlay follows issue #36452's guarded lifecycle proposal so the
temporary BF16 draft embedding and head are rebound to target weights before
target memory-pool sizing.

## Pinned build and identities

Run these on the canonical `.177` source host only after budgeting the archive
and layer-cache peak. They do not use a GPU.

```bash
cd /home/aday/NexusAgentDashboard/bc_aeon/aeon/services/sglang
aeon_oci_archive=/home/aday/.local/state/aeon-flash-next/runtime-images/qwen38-flash-next-sm120-headroom-a6c61-424e.oci.tar
test ! -e "$aeon_oci_archive"
install -d -m 700 "$(dirname "$aeon_oci_archive")"
/home/aday/bin/fleet-low-priority docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --sbom=false \
  --file Dockerfile.qwen38-flash-next-sm120 \
  --output "type=oci,name=aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e,dest=$aeon_oci_archive" \
  .

aeon_manifest_digest="$(tar -xOf "$aeon_oci_archive" index.json | jq -er '.manifests | if length == 1 then .[0].digest else error("expected one image manifest") end')"
aeon_manifest_hex="${aeon_manifest_digest#sha256:}"
aeon_config_digest="$(tar -xOf "$aeon_oci_archive" "blobs/sha256/$aeon_manifest_hex" | jq -er '.config.digest')"
test "$aeon_manifest_digest" != "$aeon_config_digest"
/home/aday/bin/fleet-low-priority sha256sum "$aeon_oci_archive"
chmod 600 "$aeon_oci_archive"
/home/aday/bin/fleet-low-priority docker load --input "$aeon_oci_archive"
aeon_image_reference="aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e@$aeon_manifest_digest"
aeon_local_image_id="$(docker image inspect --format '{{.Id}}' "$aeon_image_reference")"
test "$aeon_local_image_id" = "$aeon_manifest_digest"
test "$aeon_local_image_id" != "$aeon_config_digest"
test "$(docker image inspect --format '{{.Descriptor.digest}}' "$aeon_image_reference")" = "$aeon_manifest_digest"
docker image inspect --format '{{json .Config.Labels}}' "$aeon_image_reference" | jq -S .
```

Independent post-build readback binds the headroom image to these exact
identities:

- image tag:
  `aeon/sglang:qwen38-flash-next-sm120-headroom-a6c61-424e`;
- OCI manifest and Docker 29.2/containerd local image ID:
  `sha256:067473b3134f933ebc04a3c4774b16bd400a15afcaf9eec8230c57205f7e7719`;
- raw OCI config digest, retained as non-launch provenance:
  `sha256:ac23f9a937f1e82cc1bade15079a568a73e68b1cecbe4d4f326ba330418e0a36`;
- source OCI archive SHA-256:
  `f25ab76b3f48b55e1632e020e9fc4709766bae447c42564d2058f16a4bc13374`;
- source OCI archive size: `13,951,062,528` bytes;
- composed source-stack SHA-256:
  `f9087c7d56219f49fb575c8b1008e923ddeea1ea878e46b20f8e5585317136ed`.

The earlier pre-headroom image remains comparison evidence only and is not a
qualification or launch identity. The source archive hash is release/build
evidence. The Flash-specific Fleet cache validates and transfers this same exact
OCI-layout tar; it independently
checks the archive SHA-256, OCI manifest digest, raw config digest, complete blob
closure, platform, and required labels. After loading it, the cache requires the
remote daemon's local ID and descriptor to equal the manifest digest and requires
the exact repository digest. It never routes this archive through the legacy
27B Docker-save validator. Never substitute the raw config digest for a launch
address. Before promotion, run the CPU contract tests and a legitimate
Fleet-owned daemon preflight on the selected host.

## SM120 launch contract

This overlay does not hardcode a production SGLang command. The MoE backend,
CUDA graph mode, GDN/state/replay settings, chunk size, memory fraction, and MTP
geometry are measured selector outputs, so a static example can silently drift
from the winner. The exact inner argv is copied from its measured command hash
into the generated `RUNTIME_CONFIG.json` and private model card. The Fleet
adapter then supplies and verifies the leased UUID, claim identity, 88 GiB cap,
unlimited memlock, read-only `/model` mount, cgroup limits, loopback port, and
exact repo@manifest image reference. Do not launch this image outside Fleet
Compute.

The matching main/speculative MoE backend, final NEXTN geometry, graph mode,
chunk size, and memory fraction must come from the ordered qualification
selector. `flashinfer_cutlass` is the required SM120 ModelOpt-FP4 MoE root for
both target and speculative layers. The TRTLLM-Gen MoE cubin manifest in this
runtime has no SM120 kernels and must not be forced; this is distinct from the
supported `trtllm_mha` decode-attention backend. Qualification requires at least
8 GiB physical CUDA reserve. The headroom overlay removes the eager
7,549,747,200-byte placeholder allocation; qualification must prove the physical
reduction rather than infer it. The causal MTP-off control uses the
same argv, boot isolation, prompts, and measurements but omits only
`--speculative-algorithm`, `--speculative-num-steps`,
`--speculative-eagle-topk`, and `--speculative-num-draft-tokens`. It keeps the
explicit `--speculative-draft-model-quantization unquant` argument so the BF16
MTP-weight contract cannot silently inherit target NVFP4.

Do not add `--kv-cache-dtype nvfp4`: native NEXTN plus NVFP4 KV is not currently
qualified. The `qwen3_coder` tool parser is also omitted pending resolution of
the always-thinking tool-loop issue. Raw multimodal text/image/video requests
remain part of the required qualification suite.

The 65,536-token limit above is a conservative exact-card startup baseline from
the upstream SM120 validation. Native 262,144-token startup on this 96 GB card,
with non-NVFP4 KV/state cache, has not been demonstrated and must not be claimed
without new measured evidence.
