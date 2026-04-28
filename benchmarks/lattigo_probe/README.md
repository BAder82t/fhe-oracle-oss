# Lattigo CKKS precision probe

Out-of-process Go harness that runs a CKKS circuit through Lattigo
v6 and emits decrypt-based precision statistics per input.

This is the Item 17 / Item 18 measurement probe. Python side:
[`fhe_oracle/adapters/lattigo.py`](../../fhe_oracle/adapters/lattigo.py).

## Build

```bash
cd benchmarks/lattigo_probe
go build -o lattigo_probe .
```

## Manual smoke test

```bash
echo '{"circuit":"wxb_squared","params":"n14_logq200","inputs":[[1.0,2.0,3.0]],"w":[0.1,0.2,0.3],"b":0.5}' \
  | ./lattigo_probe
# idx,mean_bits,min_bits,max_bits,std_bits,plaintext,decrypted
# 0,30.6,30.6,30.6,0.0,3.61,3.6100001
```

## Stdin schema

```json
{
  "circuit": "wxb_squared",
  "params": "n14_logq200",
  "inputs": [[x1, x2, ...], ...],
  "w": [w1, w2, ...],
  "b": 0.5
}
```

## Stdout schema (CSV, no header)

```
idx,mean_bits,min_bits,max_bits,std_bits,plaintext_value,decrypted_real
```

For Item 17 v1 the three precision columns are identical (only slot 0
inspected). Future expansion: full-slot SIMD packing → distinct mean/min/max.

## Param sets

- `n14_logq200` — LogN=14, LogQ=[55,40,40,40,40,40], LogP=[61], scale=2^40.
  ~128-bit security, depth 6, 8192 slots.

Add new sets in `paramsByName` as circuits land.

## Circuits

- `wxb_squared` — single multiplication then square: `(w·x + b)^2`.
  d=1 reduction (SIMD slot 0 only) for v1 smoke.
