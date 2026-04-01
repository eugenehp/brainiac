# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-01

### Added
- Renamed the crate from `brainiac-rs` to `brainiac`.
- Added Hugging Face model hub publication for the verified BrainIAC artifacts.
- Published a GitHub release and crates.io package under the new crate name.
- Updated the README with the latest parity and Metal benchmark results.
- Added a crates.io badge and GitHub release badge to the README.

### Changed
- Updated package metadata with `repository` and `homepage` links.
- Excluded `hf_repo/` and parity vectors from published crate artifacts.
- Refreshed numerical parity vectors against the real BrainIAC weights.

### Verified
- Full Python MONAI vs Rust parity passes with max error `4.03e-5`.
- Metal/wgpu benchmark shows ~22 ms forward time on Apple Silicon.
- Crate package verification passes with `cargo publish --dry-run`.
