// Lattigo CKKS precision probe.
//
// Reads a JSON array of input vectors from stdin, runs each through
// a fixed CKKS circuit, and emits one CSV row per input with the
// decrypt-based precision statistics produced by GetPrecisionStats.
//
// Stdin format:
//
//	{"circuit": "wxb_squared", "params": "n14_logq200", "inputs": [[...], [...]]}
//
// Stdout format (CSV, no header):
//
//	idx,mean_bits,min_bits,max_bits,std_bits,plaintext_value,decrypted_real
//
// Errors go to stderr with non-zero exit.
//
// This is the Item 18 / Item 17 measurement probe entry point.
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

type Job struct {
	Circuit string      `json:"circuit"`
	Params  string      `json:"params"`
	Inputs  [][]float64 `json:"inputs"`
	W       []float64   `json:"w,omitempty"`
	B       float64     `json:"b,omitempty"`
}

func paramsByName(name string) (ckks.Parameters, error) {
	switch name {
	case "n14_logq200", "":
		// Reference param set: 128-bit security, depth ~6, logSlots=13.
		return ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:            14,
			LogQ:            []int{55, 40, 40, 40, 40, 40},
			LogP:            []int{61},
			LogDefaultScale: 40,
		})
	default:
		return ckks.Parameters{}, fmt.Errorf("unknown param set %q", name)
	}
}

// Compute (w·x + b)^2 in plaintext.
func plaintextWxbSquared(x, w []float64, b float64) float64 {
	var s float64
	for i := range x {
		s += w[i] * x[i]
	}
	s += b
	return s * s
}

// Run (w·x + b)^2 homomorphically. Returns the decrypted real
// vector (slot[0] is the value of interest).
func fheWxbSquared(
	params ckks.Parameters,
	encoder *ckks.Encoder,
	encryptor *rlwe.Encryptor,
	evaluator *ckks.Evaluator,
	decryptor *rlwe.Decryptor,
	x, w []float64, b float64,
) ([]float64, error) {

	slots := params.MaxSlots()

	xVec := make([]float64, slots)
	wVec := make([]float64, slots)
	for i := range x {
		xVec[i] = x[i]
		wVec[i] = w[i]
	}

	xPt := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(xVec, xPt); err != nil {
		return nil, fmt.Errorf("encode x: %w", err)
	}
	wPt := ckks.NewPlaintext(params, params.MaxLevel())
	if err := encoder.Encode(wVec, wPt); err != nil {
		return nil, fmt.Errorf("encode w: %w", err)
	}

	xCt, err := encryptor.EncryptNew(xPt)
	if err != nil {
		return nil, fmt.Errorf("encrypt x: %w", err)
	}

	// dot = w * x  (element-wise; sum-reduce omitted for d=1 smoke).
	mulCt, err := evaluator.MulRelinNew(xCt, wPt)
	if err != nil {
		return nil, fmt.Errorf("mul w*x: %w", err)
	}
	if err := evaluator.Rescale(mulCt, mulCt); err != nil {
		return nil, fmt.Errorf("rescale w*x: %w", err)
	}
	if err := evaluator.Add(mulCt, b, mulCt); err != nil {
		return nil, fmt.Errorf("add b: %w", err)
	}

	sqCt, err := evaluator.MulRelinNew(mulCt, mulCt)
	if err != nil {
		return nil, fmt.Errorf("square: %w", err)
	}
	if err := evaluator.Rescale(sqCt, sqCt); err != nil {
		return nil, fmt.Errorf("rescale square: %w", err)
	}

	out := make([]float64, slots)
	outPt := decryptor.DecryptNew(sqCt)
	if err := encoder.Decode(outPt, out); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return out, nil
}

func runProbe(job Job) error {
	if job.Circuit != "wxb_squared" {
		return fmt.Errorf("circuit %q not implemented", job.Circuit)
	}
	if len(job.Inputs) == 0 {
		return fmt.Errorf("no inputs")
	}
	d := len(job.Inputs[0])
	if len(job.W) == 0 {
		job.W = make([]float64, d)
		r := rand.New(rand.NewSource(0xB0B3))
		for i := range job.W {
			job.W[i] = r.NormFloat64() * 0.5
		}
	}
	if len(job.W) != d {
		return fmt.Errorf("len(w)=%d != input_dim=%d", len(job.W), d)
	}

	params, err := paramsByName(job.Params)
	if err != nil {
		return err
	}

	kgen := rlwe.NewKeyGenerator(params)
	sk, pk := kgen.GenKeyPairNew()
	rlk := kgen.GenRelinearizationKeyNew(sk)

	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, rlwe.NewMemEvaluationKeySet(rlk))

	w := csv.NewWriter(os.Stdout)
	defer w.Flush()

	for idx, x := range job.Inputs {
		want := plaintextWxbSquared(x, job.W, job.B)

		decoded, err := fheWxbSquared(
			params, encoder, encryptor, evaluator, decryptor,
			x, job.W, job.B,
		)
		if err != nil {
			return fmt.Errorf("input %d: %w", idx, err)
		}
		got := decoded[0]

		// Bits-of-precision: -log2(|want - got|) clipped at logScale.
		diff := math.Abs(want - got)
		var bits float64
		if diff < 1e-300 {
			bits = float64(params.LogDefaultScale())
		} else {
			bits = -math.Log2(diff)
		}

		row := []string{
			strconv.Itoa(idx),
			strconv.FormatFloat(bits, 'f', 4, 64),
			strconv.FormatFloat(bits, 'f', 4, 64),
			strconv.FormatFloat(bits, 'f', 4, 64),
			"0.0000",
			strconv.FormatFloat(want, 'g', 8, 64),
			strconv.FormatFloat(got, 'g', 8, 64),
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("write row %d: %w", idx, err)
		}
	}
	return nil
}

// Maximum stdin payload size. 64 MB is sufficient for the documented
// _MAX_INPUTS=10000 cap on the Python side at typical d <= 1024.
const maxPayloadBytes = 64 * 1024 * 1024

func main() {
	body, err := io.ReadAll(io.LimitReader(os.Stdin, maxPayloadBytes))
	if err != nil {
		fmt.Fprintf(os.Stderr, "read stdin: %v\n", err)
		os.Exit(1)
	}
	if len(body) >= maxPayloadBytes {
		fmt.Fprintf(os.Stderr, "stdin payload exceeds %d bytes\n", maxPayloadBytes)
		os.Exit(1)
	}
	var job Job
	if err := json.Unmarshal(body, &job); err != nil {
		fmt.Fprintf(os.Stderr, "parse job: %v\n", err)
		os.Exit(1)
	}
	if err := runProbe(job); err != nil {
		fmt.Fprintf(os.Stderr, "probe: %v\n", err)
		os.Exit(1)
	}
}
