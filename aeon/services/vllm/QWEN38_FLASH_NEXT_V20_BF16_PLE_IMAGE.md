# Qwen3.8 Flash-Next v20 BF16-PLE attestation image

This is a CPU-built runtime-attestation overlay for the disabled v20 canary. It
starts from the immutable upstream vLLM index
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
the selected linux/amd64 child is
`sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`.

The overlay applies only:

- `patches/qwen38-flash-next-runtime-attestation.patch`;
- `qwen38_flash_next_attestation.py`.

It does not copy or apply `qwen38-flash-next-radixark-ple-fp8.patch`. The mazinb
artifact has a BF16 PLE n-gram table even though its inherited `config.json`
retains `ple_embedding_dtype=float8_e4m3fn`. Runtime tensor-dtype attestation is
authoritative; `VLLM_PLE_FP8_CHECKPOINT` must remain unset.

The Dockerfile fail-closes on upstream and patched GPU-worker/PLE-worker hashes,
requires the upstream PLE layer to remain byte-identical, and verifies vLLM
`0.1.dev20073+g8e685d198`.

Build from `bc_aeon` after the immutable upstream digest is present locally:

```bash
/home/aday/bin/fleet-low-priority /usr/bin/docker build \
  --pull=false --network=none --platform=linux/amd64 \
  --file aeon/services/vllm/Dockerfile.qwen38-flash-next-v20-bf16-ple-attestation \
  --tag aeon/vllm:qwen38-flash-next-v20-bf16-ple-attested-5d4ba3b4 \
  aeon/services/vllm
```

Export without starting a container:

```bash
umask 077
/home/aday/bin/fleet-low-priority /usr/bin/docker image save \
  --output /home/aday/.local/state/aeon-flash-next/runtime-images/qwen38-flash-next-v20-bf16-ple-attested-5d4ba3b4.oci.tar \
  aeon/vllm:qwen38-flash-next-v20-bf16-ple-attested-5d4ba3b4
```

The canary profile remains disabled until the archive SHA-256, OCI manifest,
OCI config, checkpoint manifest, checkpoint file count, and all final source
hashes are inserted and independently validated.

## Immutable CPU build receipt

The network-isolated linux/amd64 build and export completed on 2026-08-27. This
receipt documents the image artifact only; it is not authorization to fill a
profile, reload Fleet, or launch the image.

- Archive: `qwen38-flash-next-v20-bf16-ple-attested-5d4ba3b4.oci.tar`
- Archive size: `8634628608` bytes; required mode: `0600`
- Archive SHA-256: `320722f344465b162d6277e65a9a1b27eb70c9b7960259604e32da10899f4a75`
- OCI image index: `sha256:a4b571515079e107134fc866d4411f18fc1c61c3fa755e85079e357121ba13b9`
- linux/amd64 image manifest: `sha256:f1f8a4dbeb015d112a230406c22e00cf2003b1bb0377d789e5730afaf9a9cc51`
- Image config: `sha256:01285f880579e9490bd667a02769038defd00e13ab65a9981abffa3fe4943880`
- Dockerfile SHA-256: `b17d99d6708f1b4c0156764462216f382cabc9d613c9674f92064fc4ab509db0`
- Runtime patch SHA-256: `81fcf77c7a83ec177ee98010d1ace082e978567f8490fadeab48d2d71044a81e`
- Attestation module SHA-256: `5d4ba3b47dadf99e93513b7bf4663ef7b2657db082f19fa4ac038696010baf9a`

An independent network-none CPU container readback reproduced the expected
patched worker hashes and the unchanged PLE-layer hash recorded in the
Dockerfile labels. The OCI archive graph was also traversed from `index.json`;
each referenced index, manifest, and config blob matched its descriptor digest.
