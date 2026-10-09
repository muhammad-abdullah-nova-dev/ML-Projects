# High-Performance Data Preprocessing Engine (Rust & Polars)

[![Rust](https://img.shields.io/badge/Rust-1.75+-DEA584.svg?style=flat&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Polars](https://img.shields.io/badge/Polars-0.38+-CD792C.svg?style=flat&logo=polars&logoColor=white)](https://pola.rs)

A high-throughput, memory-safe data processing engine built with Rust and Polars for fast batch cleaning, null imputation, and numeric normalization of large tabular datasets.

Maintained and documented by **M. Abdullah**.

---

## ⚡ Performance Highlights

* **Memory Safety & Zero Cost Abstractions**: Zero garbage collection pauses, low memory overhead, and thread-safe data transformations.
* **Polars Apache Arrow Core**: Utilizes columnar Arrow memory layouts and multi-threaded SIMD execution for orders-of-magnitude faster processing than standard Python loops.
* **Pipeline Structure**:
  - `src/io.rs`: Buffered CSV stream reader and writer.
  - `src/schema.rs`: Type-safe schema definition and validation.
  - `src/pipeline.rs`: Null imputation and Z-score numeric normalization.
  - `src/main.rs`: High-performance execution entrypoint.

---

## 🚀 How to Run

### Prerequisites
- [Rust Toolchain (cargo & rustc)](https://rustup.rs)

### Build and Execute
```bash
cd rust-data-preprocessing
cargo build --release
cargo run
```
Reads `data/sample.csv`, cleans missing values, normalizes numeric features, and outputs `data/cleaned.csv`.

---

## 👨‍💻 Maintainer & Attribution
- **Maintainer**: **M. Abdullah**
- **Original Project Origin**: Derivative work based on open-source project by `torresjchristopher`.
- **Copyright**: Copyright © 2026 M. Abdullah for modifications.
