/**
 * Reed-Solomon Coding over 8-bit values.
 *
 * Copyright 2015, Klaus Post
 * Copyright 2015, Backblaze, Inc.
 */

// Package reedsolomon enables Erasure Coding in Go
//
// For usage and examples, see https://github.com/klauspost/reedsolomon
//
package reedsolomon

import (
	"bytes"
	"errors"
	"io"
	"runtime"
	"sync"

	//"fmt"
	"github.com/klauspost/cpuid/v2"
)

// Encoder is an interface to encode Reed-Salomon parity sets for your data.
type Encoder interface {
	// Encode parity for a set of data shards.
	// Input is 'shards' containing data shards followed by parity shards.
	// The number of shards must match the number given to New().
	// Each shard is a byte array, and they must all be the same size.
	// The parity shards will always be overwritten and the data shards
	// will remain the same, so it is safe for you to read from the
	// data shards while this is running.
	Encode(shards [][]byte) error

	// Verify returns true if the parity shards contain correct data.
	// The data is the same format as Encode. No data is modified, so
	// you are allowed to read from data while this is running.
	Verify(shards [][]byte) (bool, error)

	// Reconstruct will recreate the missing shards if possible.
	//
	// Given a list of shards, some of which contain data, fills in the
	// ones that don't have data.
	//
	// The length of the array must be equal to the total number of shards.
	// You indicate that a shard is missing by setting it to nil or zero-length.
	// If a shard is zero-length but has sufficient capacity, that memory will
	// be used, otherwise a new []byte will be allocated.
	//
	// If there are too few shards to reconstruct the missing
	// ones, ErrTooFewShards will be returned.
	//
	// The reconstructed shard set is complete, but integrity is not verified.
	// Use the Verify function to check if data set is ok.
	Reconstruct(shards [][]byte) error

	// ReconstructData will recreate any missing data shards, if possible.
	//
	// Given a list of shards, some of which contain data, fills in the
	// data shards that don't have data.
	//
	// The length of the array must be equal to Shards.
	// You indicate that a shard is missing by setting it to nil or zero-length.
	// If a shard is zero-length but has sufficient capacity, that memory will
	// be used, otherwise a new []byte will be allocated.
	//
	// If there are too few shards to reconstruct the missing
	// ones, ErrTooFewShards will be returned.
	//
	// As the reconstructed shard set may contain missing parity shards,
	// calling the Verify function is likely to fail.
	ReconstructData(shards [][]byte) error

	// Update parity is use for change a few data shards and update it's parity.
	// Input 'newDatashards' containing data shards changed.
	// Input 'shards' containing old data shards (if data shard not changed, it can be nil) and old parity shards.
	// new parity shards will in shards[DataShards:]
	// Update is very useful if  DataShards much larger than ParityShards and changed data shards is few. It will
	// faster than Encode and not need read all data shards to encode.
	Update(shards [][]byte, newDatashards [][]byte) error

	// Split a data slice into the number of shards given to the encoder,
	// and create empty parity shards.
	//
	// The data will be split into equally sized shards.
	// If the data size isn't dividable by the number of shards,
	// the last shard will contain extra zeros.
	//
	// There must be at least 1 byte otherwise ErrShortData will be
	// returned.
	//
	// The data will not be copied, except for the last shard, so you
	// should not modify the data of the input slice afterwards.
	Split(data []byte) ([][]byte, error)

	// Join the shards and write the data segment to dst.
	//
	// Only the data shards are considered.
	// You must supply the exact output size you want.
	// If there are to few shards given, ErrTooFewShards will be returned.
	// If the total data size is less than outSize, ErrShortData will be returned.
	Join(dst io.Writer, shards [][]byte, outSize int) error
}

// reedSolomon contains a matrix for a specific
// distribution of datashards and parity shards.
// Construct if using New()
type reedSolomon struct {
	DataShards   int // Number of data shards, should not be modified.
	ParityShards int // Number of parity shards, should not be modified.
	Shards       int // Total number of shards. Calculated, and should not be modified.
	m            matrix
	tree         *inversionTree
	parity       [][]byte
	o            options
	mPool        sync.Pool
}

// ErrInvShardNum will be returned by New, if you attempt to create
// an Encoder with less than one data shard or less than zero parity
// shards.
var ErrInvShardNum = errors.New("cannot create Encoder with less than one data shard or less than zero parity shards")

// ErrMaxShardNum will be returned by New, if you attempt to create an
// Encoder where data and parity shards are bigger than the order of
// GF(2^8).
var ErrMaxShardNum = errors.New("cannot create Encoder with more than 256 data+parity shards")

// buildMatrix creates the matrix to use for encoding, given the
// number of data shards and the number of total shards.
//
// The top square of the matrix is guaranteed to be an identity
// matrix, which means that the data shards are unchanged after
// encoding.
/*
给定数据分片的数量和总分片的数量，buildMatrix创建用于编码的矩阵。
保证矩阵的顶部是一个单位矩阵，这意味着编码后数据分片保持不变。
范德蒙矩阵
 */
func buildMatrix(dataShards, totalShards int) (matrix, error) {
	// Start with a Vandermonde matrix.  This matrix would work,
	// in theory, but doesn't have the property that the data
	// shards are unchanged after encoding.
	vm, err := vandermonde(totalShards, dataShards)
	if err != nil {
		return nil, err
	}

	// Multiply by the inverse of the top square of the matrix.
	// This will make the top square be the identity matrix, but
	// preserve the property that any square subset of rows is
	// invertible.
	top, err := vm.SubMatrix(0, 0, dataShards, dataShards)
	if err != nil {
		return nil, err
	}

	topInv, err := top.Invert()
	if err != nil {
		return nil, err
	}
	//fmt.Println(vm.Multiply(topInv))
	codeMatrix, e := vm.Multiply(topInv)

	for i := dataShards; i< dataShards + 2;i ++{
		for j:=0; j< dataShards;j++{
			if(i == dataShards && j < dataShards / 2){
				codeMatrix[i][j]  = 1;
			}else if(i == dataShards && j >= dataShards / 2){
				codeMatrix[i][j] = 0;
			}else if(i == dataShards + 1 && j < dataShards /2){
				codeMatrix[i][j]  = 0;
			}else if(i == dataShards + 1 && j >= dataShards /2){
				codeMatrix[i][j]  = 1;
			}
		}
	}
	//fmt.Println(codeMatrix)
	return codeMatrix,e
}

// buildMatrixPAR1 creates the matrix to use for encoding according to
// the PARv1 spec, given the number of data shards and the number of
// total shards. Note that the method they use is buggy, and may lead
// to cases where recovery is impossible, even if there are enough
// parity shards.
//
// The top square of the matrix is guaranteed to be an identity
// matrix, which means that the data shards are unchanged after
// encoding.
func buildMatrixPAR1(dataShards, totalShards int) (matrix, error) {
	result, err := newMatrix(totalShards, dataShards)
	if err != nil {
		return nil, err
	}

	for r, row := range result {
		// The top portion of the matrix is the identity
		// matrix, and the bottom is a transposed Vandermonde
		// matrix starting at 1 instead of 0.
		if r < dataShards {
			result[r][r] = 1
		} else {
			for c := range row {
				result[r][c] = galExp(byte(c+1), r-dataShards)
			}
		}
	}
	return result, nil
}

func buildMatrixCauchy(dataShards, totalShards int) (matrix, error) {
	result, err := newMatrix(totalShards, dataShards)
	if err != nil {
		return nil, err
	}

	for r, row := range result {
		// The top portion of the matrix is the identity
		// matrix, and the bottom is a transposed Cauchy matrix.
		if r < dataShards {
			result[r][r] = 1
		} else {
			for c := range row {
				result[r][c] = invTable[(byte(r ^ c))]
			}
		}
	}
	return result, nil
}

// buildXorMatrix can be used to build a matrix with pure XOR
// operations if there is only one parity shard.
func buildXorMatrix(dataShards, totalShards int) (matrix, error) {
	if dataShards+1 != totalShards {
		return nil, errors.New("internal error")
	}
	result, err := newMatrix(totalShards, dataShards)
	if err != nil {
		return nil, err
	}

	for r, row := range result {
		// The top portion of the matrix is the identity
		// matrix.
		if r < dataShards {
			result[r][r] = 1
		} else {
			// Set all values to 1 (XOR)
			for c := range row {
				result[r][c] = 1
			}
		}
	}
	return result, nil
}

// New creates a new encoder and initializes it to
// the number of data shards and parity shards that
// you want to use. You can reuse this encoder.
// Note that the maximum number of total shards is 256.
// If no options are supplied, default options are used.
func New(dataShards, parityShards int, opts ...Option) (Encoder, error) {
	r := reedSolomon{
		DataShards:   dataShards,
		ParityShards: parityShards,
		Shards:       dataShards + parityShards,
		o:            defaultOptions,
	}

	for _, opt := range opts {
		opt(&r.o)
	}
	if dataShards <= 0 || parityShards < 0 {
		return nil, ErrInvShardNum
	}

	if dataShards+parityShards > 256 {
		return nil, ErrMaxShardNum
	}

	if parityShards == 0 {
		return &r, nil
	}

	var err error
	//switch {
	//case r.o.fastOneParity && parityShards == 1:
	//	r.m, err = buildXorMatrix(dataShards, r.Shards)
	//case r.o.useCauchy:
	//	r.m, err = buildMatrixCauchy(dataShards, r.Shards)
	//case r.o.usePAR1Matrix:
	//	r.m, err = buildMatrixPAR1(dataShards, r.Shards)
	//default:
	//	r.m, err = buildMatrix(dataShards, r.Shards)
	//}

	// 全部都改成buildMatrix？
	r.m, err = buildMatrix(dataShards,r.Shards)

	if err != nil {
		return nil, err
	}

	// Calculate what we want per round
	r.o.perRound = cpuid.CPU.Cache.L2
	if r.o.perRound <= 0 {
		// Set to 128K if undetectable.
		r.o.perRound = 128 << 10
	}

	if cpuid.CPU.ThreadsPerCore > 1 && r.o.maxGoroutines > cpuid.CPU.PhysicalCores {
		// If multiple threads per core, make sure they don't contend for cache.
		r.o.perRound /= cpuid.CPU.ThreadsPerCore
	}
	// 1 input + parity must fit in cache, and we add one more to be safer.
	r.o.perRound = r.o.perRound / (1 + parityShards)
	// Align to 64 bytes.
	r.o.perRound = ((r.o.perRound + 63) / 64) * 64

	if r.o.minSplitSize <= 0 {
		// Set minsplit as high as we can, but still have parity in L1.
		cacheSize := cpuid.CPU.Cache.L1D
		if cacheSize <= 0 {
			cacheSize = 32 << 10
		}

		r.o.minSplitSize = cacheSize / (parityShards + 1)
		// Min 1K
		if r.o.minSplitSize < 1024 {
			r.o.minSplitSize = 1024
		}
	}

	if r.o.perRound < r.o.minSplitSize {
		r.o.perRound = r.o.minSplitSize
	}

	if r.o.shardSize > 0 {
		p := runtime.GOMAXPROCS(0)
		if p == 1 || r.o.shardSize <= r.o.minSplitSize*2 {
			// Not worth it.
			r.o.maxGoroutines = 1
		} else {
			g := r.o.shardSize / r.o.perRound

			// Overprovision by a factor of 2.
			if g < p*2 && r.o.perRound > r.o.minSplitSize*2 {
				g = p * 2
				r.o.perRound /= 2
			}

			// Have g be multiple of p
			g += p - 1
			g -= g % p

			r.o.maxGoroutines = g
		}
	}

	// Inverted matrices are cached in a tree keyed by the indices
	// of the invalid rows of the data to reconstruct.
	// The inversion root node will have the identity matrix as
	// its inversion matrix because it implies there are no errors
	// with the original data.
	// 反转矩阵被缓存在一棵树上，这棵树是由要重建的数据的无效行的索引所决定的。
	// 反转根节点将以单位矩阵作为其反转矩阵，因为这意味着原始数据不存在错误。
	if r.o.inversionCache {
		r.tree = newInversionTree(dataShards, parityShards)
	}

	r.parity = make([][]byte, parityShards)
	for i := range r.parity {
		r.parity[i] = r.m[dataShards+i]
	}

	if avx2CodeGen && r.o.useAVX2 {
		r.mPool.New = func() interface{} {
			return make([]byte, r.Shards*2*32)
		}
	}
	return &r, err
}

// ErrTooFewShards is returned if too few shards where given to
// Encode/Verify/Reconstruct/Update. It will also be returned from Reconstruct
// if there were too few shards to reconstruct the missing data.
var ErrTooFewShards = errors.New("too few shards given")

// Encodes parity for a set of data shards.
// An array 'shards' containing data shards followed by parity shards.
// The number of shards must match the number given to New.
// Each shard is a byte array, and they must all be the same size.
// The parity shards will always be overwritten and the data shards
// will remain the same.
func (r *reedSolomon) Encode(shards [][]byte) error {
	if len(shards) != r.Shards {
		return ErrTooFewShards
	}

	err := checkShards(shards, false)
	if err != nil {
		return err
	}

	// Get the slice of output buffers.
	//output是编码后的矩阵？（即不包含原数据矩阵？）
	output := shards[r.DataShards:]

	// Do the coding.
	// parity 是编码矩阵的非单位矩阵部分，shard[0:r.DataShards]是编码后的数据矩阵（未改变），output，ParityShards是编码后矩阵的行数，len是列数
	r.codeSomeShards(r.parity, shards[0:r.DataShards], output, r.ParityShards, len(shards[0]))
	return nil
}

// ErrInvalidInput is returned if invalid input parameter of Update.
var ErrInvalidInput = errors.New("invalid input")

func (r *reedSolomon) Update(shards [][]byte, newDatashards [][]byte) error {
	if len(shards) != r.Shards {
		return ErrTooFewShards
	}

	if len(newDatashards) != r.DataShards {
		return ErrTooFewShards
	}

	err := checkShards(shards, true)
	if err != nil {
		return err
	}

	err = checkShards(newDatashards, true)
	if err != nil {
		return err
	}

	for i := range newDatashards {
		if newDatashards[i] != nil && shards[i] == nil {
			return ErrInvalidInput
		}
	}
	for _, p := range shards[r.DataShards:] {
		if p == nil {
			return ErrInvalidInput
		}
	}

	shardSize := shardSize(shards)

	// Get the slice of output buffers.
	output := shards[r.DataShards:]

	// Do the coding.
	r.updateParityShards(r.parity, shards[0:r.DataShards], newDatashards[0:r.DataShards], output, r.ParityShards, shardSize)
	return nil
}

func (r *reedSolomon) updateParityShards(matrixRows, oldinputs, newinputs, outputs [][]byte, outputCount, byteCount int) {
	if len(outputs) == 0 {
		return
	}

	if r.o.maxGoroutines > 1 && byteCount > r.o.minSplitSize {
		r.updateParityShardsP(matrixRows, oldinputs, newinputs, outputs, outputCount, byteCount)
		return
	}

	for c := 0; c < r.DataShards; c++ {
		in := newinputs[c]
		if in == nil {
			continue
		}
		oldin := oldinputs[c]
		// oldinputs data will be change
		sliceXor(in, oldin, &r.o)
		for iRow := 0; iRow < outputCount; iRow++ {
			galMulSliceXor(matrixRows[iRow][c], oldin, outputs[iRow], &r.o)
		}
	}
}

func (r *reedSolomon) updateParityShardsP(matrixRows, oldinputs, newinputs, outputs [][]byte, outputCount, byteCount int) {
	var wg sync.WaitGroup
	do := byteCount / r.o.maxGoroutines
	if do < r.o.minSplitSize {
		do = r.o.minSplitSize
	}
	start := 0
	for start < byteCount {
		if start+do > byteCount {
			do = byteCount - start
		}
		wg.Add(1)
		go func(start, stop int) {
			for c := 0; c < r.DataShards; c++ {
				in := newinputs[c]
				if in == nil {
					continue
				}
				oldin := oldinputs[c]
				// oldinputs data will be change
				sliceXor(in[start:stop], oldin[start:stop], &r.o)
				for iRow := 0; iRow < outputCount; iRow++ {
					galMulSliceXor(matrixRows[iRow][c], oldin[start:stop], outputs[iRow][start:stop], &r.o)
				}
			}
			wg.Done()
		}(start, start+do)
		start += do
	}
	wg.Wait()
}

// Verify returns true if the parity shards contain the right data.
// The data is the same format as Encode. No data is modified.
func (r *reedSolomon) Verify(shards [][]byte) (bool, error) {
	if len(shards) != r.Shards {
		return false, ErrTooFewShards
	}
	err := checkShards(shards, false)
	if err != nil {
		return false, err
	}

	// Slice of buffers being checked.
	toCheck := shards[r.DataShards:]

	// Do the checking.
	// parity 是编码矩阵的非单位矩阵部分，shard[0:r.DataShards]是编码后的数据矩阵（未改变），tocheck是编码后的矩阵部分，ParityShards是编码后矩阵的行数，len是列数
	return r.checkSomeShards(r.parity, shards[0:r.DataShards], toCheck, r.ParityShards, len(shards[0])), nil
}

// Multiplies a subset of rows from a coding matrix by a full set of
// input shards to produce some output shards.
// 'matrixRows' is The rows from the matrix to use.
// 'inputs' An array of byte arrays, each of which is one input shard.
// The number of inputs used is determined by the length of each matrix row.
// outputs Byte arrays where the computed shards are stored.
// The number of outputs computed, and the
// number of matrix rows used, is determined by
// outputCount, which is the number of outputs to compute.
//将编码矩阵中的行子集乘以完整的输入分片集，以生成一些输出分片。
//'matrixRows'是要使用的矩阵中的行。
//'inputs'字节数组的数组，每个字节数组都是一个输入分片。
//使用的输入数量取决于每个矩阵行的长度。
//输出字节数组，其中存储了计算出的碎片。
//计算的输出数量以及所使用的矩阵行的数量由outputCount决定，outputCount是要计算的输出数量。
// parity 是编码矩阵的非单位矩阵部分，shard[0:r.DataShards]是编码后的数据矩阵（未改变），output，ParityShards是编码后矩阵的行数，len是列数
//checkSomechards要改嘛？
func (r *reedSolomon) codeSomeShards(matrixRows, inputs, outputs [][]byte, outputCount, byteCount int) {
	if len(outputs) == 0 {
		return
	}

	//这里不能使用多线程
	//switch {
	//case r.o.useAVX512 && r.o.maxGoroutines > 1 && byteCount > r.o.minSplitSize && len(inputs) >= 4 && len(outputs) >= 2:
	//	r.codeSomeShardsAvx512P(matrixRows, inputs, outputs, outputCount, byteCount)
	//	return
	//case r.o.useAVX512 && len(inputs) >= 4 && len(outputs) >= 2:
	//	r.codeSomeShardsAvx512(matrixRows, inputs, outputs, outputCount, byteCount)
	//	return
	//case r.o.maxGoroutines > 1 && byteCount > r.o.minSplitSize:
	//	r.codeSomeShardsP(matrixRows, inputs, outputs, outputCount, byteCount)
	//	return
	//}

	// Process using no goroutines
	start, end := 0, r.o.perRound
	if end > len(inputs[0]) {
		end = len(inputs[0])
	}
	end = len(inputs[0])

	//if avx2CodeGen && r.o.useAVX2 && byteCount >= 32 && len(inputs)+len(outputs) >= 4 && len(inputs) <= maxAvx2Inputs && len(outputs) <= maxAvx2Outputs {
	//	m := genAvx2Matrix(matrixRows, len(inputs), len(outputs), r.mPool.Get().([]byte))
	//	start += galMulSlicesAvx2(m, inputs, outputs, 0, byteCount)
	//	r.mPool.Put(m)
	//	end = len(inputs[0])
	//}

	for start < len(inputs[0]) {
		for c := 0; c < r.DataShards; c++ {
			in := inputs[c][start:end]
			for iRow := 0; iRow < outputCount; iRow++ {
				if c == 0 {
					galMulSlice(matrixRows[iRow][c], in, outputs[iRow][start:end], &r.o)
				} else {
					galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow][start:end], &r.o)
				}
			}
		}
		start = end
		end += r.o.perRound
		if end > len(inputs[0]) {
			end = len(inputs[0])
		}
		end = len(inputs[0])
	}
}

// Perform the same as codeSomeShards, but split the workload into
// several goroutines.
func (r *reedSolomon) codeSomeShardsP(matrixRows, inputs, outputs [][]byte, outputCount, byteCount int) {
	var wg sync.WaitGroup
	gor := r.o.maxGoroutines

	var avx2Matrix []byte
	useAvx2 := avx2CodeGen && r.o.useAVX2 && byteCount >= 32 && len(inputs)+len(outputs) >= 4 && len(inputs) <= maxAvx2Inputs && len(outputs) <= maxAvx2Outputs
	if useAvx2 {
		avx2Matrix = genAvx2Matrix(matrixRows, len(inputs), len(outputs), r.mPool.Get().([]byte))
		defer r.mPool.Put(avx2Matrix)
	}

	do := byteCount / gor
	if do < r.o.minSplitSize {
		do = r.o.minSplitSize
	}

	// Make sizes divisible by 64
	do = (do + 63) & (^63)
	start := 0
	for start < byteCount {
		if start+do > byteCount {
			do = byteCount - start
		}

		wg.Add(1)
		go func(start, stop int) {
			if useAvx2 && stop-start >= 32 {
				start += galMulSlicesAvx2(avx2Matrix, inputs, outputs, start, stop)
			}

			lstart, lstop := start, start+r.o.perRound
			if lstop > stop {
				lstop = stop
			}
			for lstart < stop {
				for c := 0; c < r.DataShards; c++ {
					in := inputs[c][lstart:lstop]
					for iRow := 0; iRow < outputCount; iRow++ {
						if c == 0 {
							galMulSlice(matrixRows[iRow][c], in, outputs[iRow][lstart:lstop], &r.o)
						} else {
							galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow][lstart:lstop], &r.o)
						}
					}
				}
				lstart = lstop
				lstop += r.o.perRound
				if lstop > stop {
					lstop = stop
				}
			}
			wg.Done()
		}(start, start+do)
		start += do
	}
	wg.Wait()
}

// checkSomeShards is mostly the same as codeSomeShards,
// except this will check values and return
// as soon as a difference is found.
// parity 是编码矩阵的非单位矩阵部分，shard[0:r.DataShards]是编码后的数据矩阵（未改变），tocheck是编码后的矩阵部分，ParityShards是编码后矩阵的行数，len是列数
func (r *reedSolomon) checkSomeShards(matrixRows, inputs, toCheck [][]byte, outputCount, byteCount int) bool {
	if len(toCheck) == 0 {
		return true
	}
	if r.o.maxGoroutines > 1 && byteCount > r.o.minSplitSize {
		return r.checkSomeShardsP(matrixRows, inputs, toCheck, outputCount, byteCount)
	}
	outputs := make([][]byte, len(toCheck))
	for i := range outputs {
		outputs[i] = make([]byte, byteCount)
	}
	for c := 0; c < r.DataShards; c++ {
		in := inputs[c]
		for iRow := 0; iRow < outputCount; iRow++ {
			galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow], &r.o)
		}
	}

	for i, calc := range outputs {
		if !bytes.Equal(calc, toCheck[i]) {
			return false
		}
	}
	return true
}

func (r *reedSolomon) checkSomeShardsP(matrixRows, inputs, toCheck [][]byte, outputCount, byteCount int) bool {
	same := true
	var mu sync.RWMutex // For above

	var wg sync.WaitGroup
	do := byteCount / r.o.maxGoroutines
	if do < r.o.minSplitSize {
		do = r.o.minSplitSize
	}
	// Make sizes divisible by 64
	do = (do + 63) & (^63)
	start := 0
	for start < byteCount {
		if start+do > byteCount {
			do = byteCount - start
		}
		wg.Add(1)
		go func(start, do int) {
			defer wg.Done()
			outputs := make([][]byte, len(toCheck))
			for i := range outputs {
				outputs[i] = make([]byte, do)
			}
			for c := 0; c < r.DataShards; c++ {
				mu.RLock()
				if !same {
					mu.RUnlock()
					return
				}
				mu.RUnlock()
				in := inputs[c][start : start+do]
				for iRow := 0; iRow < outputCount; iRow++ {
					galMulSliceXor(matrixRows[iRow][c], in, outputs[iRow], &r.o)
				}
			}

			for i, calc := range outputs {
				if !bytes.Equal(calc, toCheck[i][start:start+do]) {
					mu.Lock()
					same = false
					mu.Unlock()
					return
				}
			}
		}(start, do)
		start += do
	}
	wg.Wait()
	return same
}

// ErrShardNoData will be returned if there are no shards,
// or if the length of all shards is zero.
var ErrShardNoData = errors.New("no shard data")

// ErrShardSize is returned if shard length isn't the same for all
// shards.
var ErrShardSize = errors.New("shard sizes do not match")

// checkShards will check if shards are the same size
// or 0, if allowed. An error is returned if this fails.
// An error is also returned if all shards are size 0.
func checkShards(shards [][]byte, nilok bool) error {
	size := shardSize(shards)
	if size == 0 {
		return ErrShardNoData
	}
	for _, shard := range shards {
		if len(shard) != size {
			if len(shard) != 0 || !nilok {
				return ErrShardSize
			}
		}
	}
	return nil
}

// shardSize return the size of a single shard.
// The first non-zero size is returned,
// or 0 if all shards are size 0.
func shardSize(shards [][]byte) int {
	for _, shard := range shards {
		if len(shard) != 0 {
			return len(shard)
		}
	}
	return 0
}

// Reconstruct will recreate the missing shards, if possible.
//
// Given a list of shards, some of which contain data, fills in the
// ones that don't have data.
//
// The length of the array must be equal to Shards.
// You indicate that a shard is missing by setting it to nil or zero-length.
// If a shard is zero-length but has sufficient capacity, that memory will
// be used, otherwise a new []byte will be allocated.
//
// If there are too few shards to reconstruct the missing
// ones, ErrTooFewShards will be returned.
//
// The reconstructed shard set is complete, but integrity is not verified.
// Use the Verify function to check if data set is ok.
func (r *reedSolomon) Reconstruct(shards [][]byte) error {
	// fmt.Println("myRS")
	return r.reconstruct(shards, false)
}

// ReconstructData will recreate any missing data shards, if possible.
//
// Given a list of shards, some of which contain data, fills in the
// data shards that don't have data.
//
// The length of the array must be equal to Shards.
// You indicate that a shard is missing by setting it to nil or zero-length.
// If a shard is zero-length but has sufficient capacity, that memory will
// be used, otherwise a new []byte will be allocated.
//
// If there are too few shards to reconstruct the missing
// ones, ErrTooFewShards will be returned.
//
// As the reconstructed shard set may contain missing parity shards,
// calling the Verify function is likely to fail.
func (r *reedSolomon) ReconstructData(shards [][]byte) error {
	return r.reconstruct(shards, true)
}

// reconstruct will recreate the missing data shards, and unless
// dataOnly is true, also the missing parity shards
//
// The length of the array must be equal to Shards.
// You indicate that a shard is missing by setting it to nil.
//
// If there are too few shards to reconstruct the missing
// ones, ErrTooFewShards will be returned.
func (r *reedSolomon) reconstruct(shards [][]byte, dataOnly bool) error {
	//fmt.Println("myRS")
	if len(shards) != r.Shards {
		return ErrTooFewShards
	}
	// Check arguments.
	err := checkShards(shards, true)
	if err != nil {
		return err
	}


	//返回数据的列数？？？
	shardSize := shardSize(shards)
	//fmt.Println("shardsize: ",shardSize)

	// Quick check: are all of the shards present?  If so, there's
	// nothing to do. 即：数据没有损坏，直接返回即可
	numberPresent := 0
	dataPresent := 0
	for i := 0; i < r.Shards; i++ {
		if len(shards[i]) != 0 {
			numberPresent++
			if i < r.DataShards {
				dataPresent++
			}
		}
	}
	if numberPresent == r.Shards || dataOnly && dataPresent == r.DataShards {
		// Cool.  All of the shards data data.  We don't
		// need to do anything.
		return nil
	}

	// More complete sanity check   剩余的所有数据（数据＋编码）的行数少于真实数据部分的行数
	if numberPresent < r.DataShards {
		return ErrTooFewShards
	}

	// Pull out an array holding just the shards that
	// correspond to the rows of the submatrix.  These shards
	// will be the input to the decoding process that re-creates
	// the missing data shards.
	//
	// Also, create an array of indices of the valid rows we do have
	// and the invalid rows we don't have up until we have enough valid rows.
	//拉出一个仅保留与子矩阵的行相对应的分片的数组。 这些碎片将作为重新创建丢失的数据碎片的解码过程的输入。
	//
	//同样，在拥有足够的有效行之前，创建一个包含我们拥有的有效行和我们没有的无效行的索引的数组。
	subShards := make([][]byte, r.DataShards)
	validIndices := make([]int, r.DataShards)
	invalidIndices := make([]int, 0)
	subMatrixRow := 0   //行的计数
	for matrixRow := 0; matrixRow < r.Shards && subMatrixRow < r.DataShards; matrixRow++ {
		if len(shards[matrixRow]) != 0 { //如果传进来的该数据行存在，则将数据的该行保存到subShards中，将该行的下标保存在validIndices中。
			subShards[subMatrixRow] = shards[matrixRow]
			validIndices[subMatrixRow] = matrixRow
			subMatrixRow++
		} else {
			invalidIndices = append(invalidIndices, matrixRow)   //如果该行不存在，则把该行的下标保存到invalidIndices中
		}
	}

	//判断错误的情况
	isParty := 0
	isfront := 0
	isblack := 0
	if len(invalidIndices) > 1 {
		for i:= 0;i < len(invalidIndices);i++ {
			if invalidIndices[i] < r.DataShards/2 {
				isfront = 1
			}
			if invalidIndices[i] >= r.DataShards / 2 && invalidIndices[i] < r.DataShards {
				isblack = 1
			}
			if invalidIndices[i] >= r.DataShards {
				isParty = 1
			}

		}
	}



	// Attempt to get the cached inverted matrix out of the tree
	// based on the indices of the invalid rows.
	////尝试根据无效行的索引从树中获取缓存的倒置矩阵。
	dataDecodeMatrix := r.tree.GetInvertedMatrix(invalidIndices)


	//custom
	//fmt.Println("invalidIndices[0]",invalidIndices[0])
	if len(invalidIndices) == 1 && invalidIndices[0] < r.DataShards {
		//fmt.Println("invalidIndeices = 1")
		subMatrix, _ := newMatrix(r.DataShards, r.DataShards)   //生成一个nxn的矩阵，n为数据部分的行数
		for subMatrixRow, validIndex := range validIndices {  //填充该矩阵的内容，这里的
			for c := 0; c < r.DataShards; c++ {
				subMatrix[subMatrixRow][c] = r.m[validIndex][c]  //将输入数据中有效行在原编码矩阵m中对应的行存放到submatrix中
			}
		}
		//fmt.Println(subMatrix)
		//对subMatrix做个处理
		//对 subShards 和 validIndeices做个处理
		if(invalidIndices[0] >= (r.DataShards)/2){
			for l:= 0;l < r.DataShards;l++{
				subMatrix[r.DataShards-1][l] = r.m[r.DataShards+1][l]
			}
			subShards[subMatrixRow-1] = shards[r.DataShards+1]
			validIndices[subMatrixRow-1] = r.DataShards+1
		}

		dataDecodeMatrix, err = subMatrix.Invert()    //还存在的行对应的编码矩阵的逆矩阵
		//fmt.Println(dataDecodeMatrix)
		if err != nil {
			return err
		}

		//outputs := make([][]byte, 1)
		//matrixRows := make([][]byte, 1)
		//outputCount := 0
		//
		//for iShard := 0; iShard < r.DataShards; iShard++ {   //取出剩余数据中真实数据部分的 被损坏的行
		//	if len(shards[iShard]) == 0 {  //当前行损坏   损坏的行数会小于n？？
		//		if cap(shards[iShard]) >= shardSize {   //为当前行分配空间
		//			shards[iShard] = shards[iShard][0:shardSize]
		//		} else {
		//			shards[iShard] = make([]byte, shardSize)
		//		}
		//		outputs[outputCount] = shards[iShard]
		//		matrixRows[outputCount] = dataDecodeMatrix[iShard]
		//		outputCount++
		//	}
		//}
		//r.codeSomeShards(matrixRows, subShards, outputs[:outputCount], outputCount, shardSize)
		//
		//if dataOnly {
		//	// Exit out early if we are only interested in the data shards
		//	return nil
		//}
		//
		//outputCount = 0
		//for iShard := r.DataShards; iShard < r.Shards; iShard++ { //
		//	if len(shards[iShard]) == 0 {  //传入的数据的已编码部分，如果被损坏
		//		if cap(shards[iShard]) >= shardSize {
		//			shards[iShard] = shards[iShard][0:shardSize]
		//		} else {
		//			shards[iShard] = make([]byte, shardSize)
		//		}
		//		outputs[outputCount] = shards[iShard]
		//		matrixRows[outputCount] = r.parity[iShard-r.DataShards]
		//		outputCount++
		//	}
		//}
		////matrixRows 编码矩阵中的非单位矩阵部分
		////shards[:r.DataShards] 数据部分
		//r.codeSomeShards(matrixRows, shards[:r.DataShards], outputs[:outputCount], outputCount, shardSize)

	} else if isfront == 1 && isblack == 0 && isParty == 0{  //缺失多行 但是都在前半部分

		//对subMatrix做个处理
		//对 subShards 和 validIndeices做个处理
		//先修改subShards,validIndeices 再修改 subMatrix
		subMatrixRow := 0   //行的计数
		for matrixRow := 0; matrixRow < r.Shards && subMatrixRow < r.DataShards; matrixRow++ {
			if len(shards[matrixRow]) != 0 && matrixRow != r.DataShards+1{ //如果传进来的该数据行存在，则将数据的该行保存到subShards中，将该行的下标保存在validIndices中。
				subShards[subMatrixRow] = shards[matrixRow]
				validIndices[subMatrixRow] = matrixRow
				subMatrixRow++
			} else if len(shards[matrixRow]) == 0 {
				invalidIndices = append(invalidIndices, matrixRow)   //如果该行不存在，则把该行的下标保存到invalidIndices中
			}
		}

		subMatrix, _ := newMatrix(r.DataShards, r.DataShards)   //生成一个nxn的矩阵，n为数据部分的行数
		for subMatrixRow1, validIndex := range validIndices {  //填充该矩阵的内容，这里的
			for c := 0; c < r.DataShards; c++ {
				subMatrix[subMatrixRow1][c] = r.m[validIndex][c]  //将输入数据中有效行在原编码矩阵m中对应的行存放到submatrix中
			}
		}

		dataDecodeMatrix, err = subMatrix.Invert()    //还存在的行对应的编码矩阵的逆矩阵
		//fmt.Println(dataDecodeMatrix)
		if err != nil {
			return err
		}


	} else if isfront == 0 && isblack == 1 && isParty == 0{
		//对subMatrix做个处理
		//对 subShards 和 validIndeices做个处理
		//先修改subShards,validIndeices 再修改 subMatrix
		subMatrixRow := 0   //行的计数
		for matrixRow := 0; matrixRow < r.Shards && subMatrixRow < r.DataShards; matrixRow++ {
			if len(shards[matrixRow]) != 0 && matrixRow != r.DataShards { //如果传进来的该数据行存在，则将数据的该行保存到subShards中，将该行的下标保存在validIndices中。
				subShards[subMatrixRow] = shards[matrixRow]
				validIndices[subMatrixRow] = matrixRow
				subMatrixRow++
			} else if len(shards[matrixRow]) == 0 {
				invalidIndices = append(invalidIndices, matrixRow)   //如果该行不存在，则把该行的下标保存到invalidIndices中
			}
		}

		subMatrix, _ := newMatrix(r.DataShards, r.DataShards)   //生成一个nxn的矩阵，n为数据部分的行数
		for subMatrixRow1, validIndex := range validIndices {  //填充该矩阵的内容，这里的
			for c := 0; c < r.DataShards; c++ {
				subMatrix[subMatrixRow1][c] = r.m[validIndex][c]  //将输入数据中有效行在原编码矩阵m中对应的行存放到submatrix中
			}
		}

		dataDecodeMatrix, err = subMatrix.Invert()    //还存在的行对应的编码矩阵的逆矩阵
		//fmt.Println(dataDecodeMatrix)
		if err != nil {
			return err
		}


	} else {



		// If the inverted matrix isn't cached in the tree yet we must
		// construct it ourselves and insert it into the tree for the
		// future.  In this way the inversion tree is lazily loaded.
		if dataDecodeMatrix == nil { //无法从树中获取其逆矩阵
			// Pull out the rows of the matrix that correspond to the
			// shards that we have and build a square matrix.  This
			// matrix could be used to generate the shards that we have
			// from the original data.
			////如果倒置矩阵未缓存在树中，则我们必须自己构造它，并将其插入树中以备将来使用。 这样，反型树被延迟加载。
			// 这里尝试构造其逆矩阵，但是不保存到树中

			subMatrix, _ := newMatrix(r.DataShards, r.DataShards) //生成一个nxn的矩阵，n为数据部分的行数
			for subMatrixRow, validIndex := range validIndices {  //填充该矩阵的内容，这里的
				for c := 0; c < r.DataShards; c++ {
					subMatrix[subMatrixRow][c] = r.m[validIndex][c] //将输入数据中有效行在原编码矩阵m中对应的行存放到submatrix中
				}
			}
			// Invert the matrix, so we can go from the encoded shards
			// back to the original data.  Then pull out the row that
			// generates the shard that we want to decode.  Note that
			// since this matrix maps back to the original data, it can
			// be used to create a data shard, but not a parity shard.
			dataDecodeMatrix, err = subMatrix.Invert() //还存在的行对应的编码矩阵的逆矩阵
			if err != nil {
				return err
			}

			// Cache the inverted matrix in the tree for future use keyed on the
			// indices of the invalid rows.
			//这里先不缓存到树中
			//err = r.tree.InsertInvertedMatrix(invalidIndices, dataDecodeMatrix, r.Shards)
			//if err != nil {
			//	return err
			//}
		}

	}
	// Re-create any data shards that were missing.
	//
	// The input to the coding is all of the shards we actually
	// have, and the output is the missing data shards.  The computation
	// is done using the special decode matrix we just built.
	//要修改的部分！！！
	//重新创建所有丢失的数据碎片。
	//编码的输入是我们实际拥有的所有分片，而输出是缺少的数据分片。 该计算是使用我们刚刚构建的特殊解码矩阵完成的。
	outputs := make([][]byte, r.ParityShards)
	matrixRows := make([][]byte, r.ParityShards)
	outputCount := 0

	for iShard := 0; iShard < r.DataShards; iShard++ { //取出剩余数据中真实数据部分的 被损坏的行
		if len(shards[iShard]) == 0 { //当前行损坏   损坏的行数会小于n？？
			if cap(shards[iShard]) >= shardSize { //为当前行分配空间
				shards[iShard] = shards[iShard][0:shardSize]
			} else {
				shards[iShard] = make([]byte, shardSize)
			}
			outputs[outputCount] = shards[iShard]
			matrixRows[outputCount] = dataDecodeMatrix[iShard]
			outputCount++
		}
	}
	//matrixRows是逆矩阵中 缺失数据对应的那几行（只包含真实数据部分，不包含编码后的部分）
	//subshards是真实数据（编码后）还存在的行
	//outputs得到是上面2个矩阵的乘积，即为真实数据的缺失部分
	r.codeSomeShards(matrixRows, subShards, outputs[:outputCount], outputCount, shardSize)

	if dataOnly {
		// Exit out early if we are only interested in the data shards
		return nil
	}

	// Now that we have all of the data shards intact, we can
	// compute any of the parity that is missing.
	//
	// The input to the coding is ALL of the data shards, including
	// any that we just calculated.  The output is whichever of the
	// data shards were missing.

	//现在我们所有数据分片都完整无缺，我们可以计算丢失的任何奇偶校验了。
	//编码的输入是所有数据分片，包括我们刚刚计算出的所有数据分片。 输出是缺少的任何数据分片。
	outputCount = 0
	for iShard := r.DataShards; iShard < r.Shards; iShard++ { //
		if len(shards[iShard]) == 0 { //传入的数据的已编码部分，如果被损坏
			if cap(shards[iShard]) >= shardSize {
				shards[iShard] = shards[iShard][0:shardSize]
			} else {
				shards[iShard] = make([]byte, shardSize)
			}
			outputs[outputCount] = shards[iShard]
			matrixRows[outputCount] = r.parity[iShard-r.DataShards]
			outputCount++
		}
	}
	//matrixRows 编码矩阵中的非单位矩阵部分
	//shards[:r.DataShards] 数据部分
	r.codeSomeShards(matrixRows, shards[:r.DataShards], outputs[:outputCount], outputCount, shardSize)

	return nil
}

// ErrShortData will be returned by Split(), if there isn't enough data
// to fill the number of shards.
var ErrShortData = errors.New("not enough data to fill the number of requested shards")

// Split a data slice into the number of shards given to the encoder,
// and create empty parity shards if necessary.
//
// The data will be split into equally sized shards.
// If the data size isn't divisible by the number of shards,
// the last shard will contain extra zeros.
//
// There must be at least 1 byte otherwise ErrShortData will be
// returned.
//
// The data will not be copied, except for the last shard, so you
// should not modify the data of the input slice afterwards.
func (r *reedSolomon) Split(data []byte) ([][]byte, error) {
	if len(data) == 0 {
		return nil, ErrShortData
	}
	// Calculate number of bytes per data shard.
	perShard := (len(data) + r.DataShards - 1) / r.DataShards

	if cap(data) > len(data) {
		data = data[:cap(data)]
	}

	// Only allocate memory if necessary
	var padding []byte
	if len(data) < (r.Shards * perShard) {
		// calculate maximum number of full shards in `data` slice
		fullShards := len(data) / perShard
		padding = make([]byte, r.Shards*perShard-perShard*fullShards)
		copy(padding, data[perShard*fullShards:])
		data = data[0 : perShard*fullShards]
	}

	// Split into equal-length shards.
	dst := make([][]byte, r.Shards)
	i := 0
	for ; i < len(dst) && len(data) >= perShard; i++ {
		dst[i] = data[:perShard:perShard]
		data = data[perShard:]
	}

	for j := 0; i+j < len(dst); j++ {
		dst[i+j] = padding[:perShard:perShard]
		padding = padding[perShard:]
	}

	return dst, nil
}

// ErrReconstructRequired is returned if too few data shards are intact and a
// reconstruction is required before you can successfully join the shards.
var ErrReconstructRequired = errors.New("reconstruction required as one or more required data shards are nil")

// Join the shards and write the data segment to dst.
//
// Only the data shards are considered.
// You must supply the exact output size you want.
//
// If there are to few shards given, ErrTooFewShards will be returned.
// If the total data size is less than outSize, ErrShortData will be returned.
// If one or more required data shards are nil, ErrReconstructRequired will be returned.
func (r *reedSolomon) Join(dst io.Writer, shards [][]byte, outSize int) error {
	// Do we have enough shards?
	if len(shards) < r.DataShards {
		return ErrTooFewShards
	}
	shards = shards[:r.DataShards]

	// Do we have enough data?
	size := 0
	for _, shard := range shards {
		if shard == nil {
			return ErrReconstructRequired
		}
		size += len(shard)

		// Do we have enough data already?
		if size >= outSize {
			break
		}
	}
	if size < outSize {
		return ErrShortData
	}

	// Copy data to dst
	write := outSize
	for _, shard := range shards {
		if write < len(shard) {
			_, err := dst.Write(shard[:write])
			return err
		}
		n, err := dst.Write(shard)
		if err != nil {
			return err
		}
		write -= n
	}
	return nil
}
