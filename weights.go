package deep

import (
	"math/rand"
	"time"
)

// A WeightInitializer returns a (random) weight
type WeightInitializer func() float64

// NewUniform returns a uniform weight generator
func NewUniform(stdDev, mean float64) WeightInitializer {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return func() float64 { return UniformRand(r, stdDev, mean) }
}

func NewUniformRand(r *rand.Rand, stdDev, mean float64) WeightInitializer {
	return func() float64 { return UniformRand(r, stdDev, mean) }
}

// Uniform samples a value from u(mean-stdDev/2,mean+stdDev/2)
func Uniform(stdDev, mean float64) float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return UniformRand(r, stdDev, mean)
}

func UniformRand(r *rand.Rand, stdDev, mean float64) float64 {
	return (r.Float64()-0.5)*stdDev + mean
}

// NewNormal returns a normal weight generator
func NewNormal(stdDev, mean float64) WeightInitializer {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return func() float64 { return NormalRand(r, stdDev, mean) }
}

func NewNormalRand(r *rand.Rand, stdDev, mean float64) WeightInitializer {
	return func() float64 { return NormalRand(r, stdDev, mean) }
}

// Normal samples a value from N(μ, σ)
func Normal(stdDev, mean float64) float64 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return NormalRand(r, stdDev, mean)
}

func NormalRand(r *rand.Rand, stdDev, mean float64) float64 {
	return r.NormFloat64()*stdDev + mean
}
