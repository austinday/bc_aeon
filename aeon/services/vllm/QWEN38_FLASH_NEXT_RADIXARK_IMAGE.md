# Qwen3.8 Flash-Next RadixArk vLLM overlay

This is a CPU-buildable, GPU-launch-free overlay for the hybrid
`RadixArk/Qwen3.8-Flash-Next-NVFP4` checkpoint. It changes only NVIDIA PLE
embedding quantization selection and scale loading. Two additional zero-fuzz
hooks emit independent post-load GPU-model and PLE CPU-worker placement
fragments. It does not replace the upstream model file, CUDA kernels,
scheduler, API server, or entrypoint.

## Bound identities

- immutable upstream index: `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
- selected linux/amd64 child: `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`
- installed vLLM: `0.1.dev20073+g8e685d198`
- upstream PLE file: `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7`
- reference implementation: `x00byte/Qwen3.8-Flash-Dual-Spark-Recipe@ba89c550dc9ff82159079c313e55fb51e1c407b5`
- PLE patch: `9fba0f454fab86efad4f13e9f836409c5d3e69cb2709275506cdfbc06d60c03f`
- patched PLE file: `fa75692d4fb941180cb4868800934cbb68a8cf26899941f5d7367d1716a90bfb`
- attestation patch: `81fcf77c7a83ec177ee98010d1ace082e978567f8490fadeab48d2d71044a81e`
- attestation module: `80459aacc387a5a70c46a3bbbb17322fe7f93e4dc30fc7e12b70da923bc34d17`
- patched GPU worker: `38820bebca30c15be82eac14f641218d5d14b8c129c1df96245da18f841817b2`
- patched PLE worker: `721a15c3440e45fd7dc41c8b5a1c441c142a1520113635ec77d78bfc833746e7`
- Dockerfile: `a2b2742a32d87070ad4521218168fa8fe1add6365345f788629a42154b07e521`
- final OCI manifest / Docker image reference: `sha256:277f87d25d8c8559849bcec660816efed1b25a3ae4701e6bbd5f815589e3af45`
- final OCI config: `sha256:185fafb00c3009615cfe1d23439da6a54f12912137c51565265839f3e8882fc5`

The Dockerfile uses the signed immutable index because Docker rejects a direct
pull of the detached child manifest. `--platform linux/amd64` selects the child
listed above, and the build fails unless both automatic platform arguments are
amd64. The child digest is also preserved in OCI labels for runtime readback.

## Rebuild

From this directory:

```bash
docker build \
  --network none \
  --platform linux/amd64 \
  --pull=false \
  --provenance=false \
  --tag aeon-vllm-qwen38-flash-next-radixark-ple-fp8:attested-80459aac \
  --file Dockerfile.qwen38-flash-next-radixark \
  .
```

The build verifies both patch SHA-256 values, every exact preimage and
postimage hash, zero-fuzz dry-run application, Python compilation, and the
installed vLLM version. The verified build uses no network and exposes no GPU.

## Canonical OCI archive receipt

The final single-platform OCI-layout archive was emitted directly from the
reviewed source through the owner's low-priority wrapper:

```bash
aeon_vllm_oci=/home/aday/.local/state/aeon-flash-next/runtime-images/qwen38-flash-next-radixark-ple-fp8-attested-80459aac.oci.tar
test ! -e "$aeon_vllm_oci"
install -d -m 700 "$(dirname "$aeon_vllm_oci")"
/home/aday/bin/fleet-low-priority docker buildx build \
  --network none \
  --platform linux/amd64 \
  --pull=false \
  --provenance=false \
  --file Dockerfile.qwen38-flash-next-radixark \
  --output "type=oci,name=aeon/vllm:qwen38-flash-next-radixark-ple-fp8-attested-80459aac,dest=$aeon_vllm_oci" \
  .
chmod 600 "$aeon_vllm_oci"
/home/aday/bin/fleet-low-priority sha256sum "$aeon_vllm_oci"
```

Readback proved:

- archive SHA-256: `f333bc334c66e18cd5b33cf7157301f774f7600c938d49aa66e2e832d94f7adf`
- archive size: `8,634,662,400` bytes
- archive mode/owner: `0600`, `aday:aday`
- exactly one linux/amd64 image with manifest and config digests listed above
- `docker load` restored exactly one tag, and
  `docker image inspect sha256:277f87d25d8c8559849bcec660816efed1b25a3ae4701e6bbd5f815589e3af45`
  succeeded with the same `.Id`.

The earlier loader-only archive remains preserved at
`qwen38-flash-next-radixark-ple-fp8-9fba0f45.oci.tar`; it is intermediate
evidence and must not be used for qualification. The generic
`qwen_artifact_cache._validate_oci_archive` currently validates legacy
Docker-save layout despite its name, so this standards-compliant OCI-layout
archive needs the canary's OCI validator rather than that legacy validator.
