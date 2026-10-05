# Changelog

## Unreleased

- Replace runtime scratch registries with native typed numerical work. Ordinary and externally supplied arrays share stage layouts and numerical bodies.
- Remove GPU Components dependencies and managed graph/workspace APIs from MJWarp. Expose native descriptor signatures, storage defaults and alignment requirements for composing engines.
- Export the native contact-force device function and vec5 type for downstream numerical reductions.
