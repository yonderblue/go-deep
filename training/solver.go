package training

import (
	"math"
)

// Solver implements an update rule for training a NN
type Solver interface {
	Init(size int)
	Step()
	Update(value, gradient float64, idx int) float64
}

// SGD is stochastic gradient descent with nesterov/momentum
type SGD struct {
	lr        float64
	decay     float64
	momentum  float64
	nesterov  bool
	moments   []float64
	iteration int
}

// NewSGD returns a new SGD solver
func NewSGD(lr, momentum, decay float64, nesterov bool) *SGD {
	return &SGD{
		lr:       fparam(lr, 0.01),
		decay:    decay,
		momentum: momentum,
		nesterov: nesterov,
	}
}

// Init initializes vectors using number of weights in network
func (o *SGD) Init(size int) {
	o.moments = make([]float64, size)
	o.iteration = 0
}

func (o *SGD) Step() {
	o.iteration++
}

// Update returns the update for a given weight
func (o *SGD) Update(value, gradient float64, idx int) float64 {
	lr := o.lr / (1 + o.decay*float64(o.iteration))

	o.moments[idx] = o.momentum*o.moments[idx] - lr*gradient

	if o.nesterov {
		o.moments[idx] = o.momentum*o.moments[idx] - lr*gradient
	}

	return o.moments[idx]
}

// Adam is an Adam solver
type Adam struct {
	lr            float64
	beta          float64
	beta2         float64
	epsilon       float64
	oneMinusBeta  float64
	oneMinusBeta2 float64

	v, m []float64

	betaPow              float64
	beta2Pow             float64
	oneMinusBetaPow      float64
	sqrtOneMinusBeta2Pow float64
	lrt                  float64
}

// NewAdam returns a new Adam solver
func NewAdam(lr, beta, beta2, epsilon float64) *Adam {
	return &Adam{
		lr:            fparam(lr, 0.001),
		beta:          fparam(beta, 0.9),
		beta2:         fparam(beta2, 0.999),
		epsilon:       fparam(epsilon, 1e-8),
		oneMinusBeta:  1 - beta,
		oneMinusBeta2: 1 - beta2,
	}
}

// Init initializes vectors using number of weights in network
func (o *Adam) Init(size int) {
	o.v, o.m = make([]float64, size), make([]float64, size)
	o.betaPow = 1
	o.beta2Pow = 1
}

func (o *Adam) Step() {
	o.betaPow *= o.beta
	o.beta2Pow *= o.beta2
	o.oneMinusBetaPow = 1.0 - o.betaPow
	o.sqrtOneMinusBeta2Pow = math.Sqrt(1.0 - o.beta2Pow)
	o.lrt = o.lr * o.sqrtOneMinusBeta2Pow / o.oneMinusBetaPow
}

// Update returns the update for a given weight
func (o *Adam) Update(value, gradient float64, idx int) float64 {
	o.m[idx] = o.beta*o.m[idx] + o.oneMinusBeta*gradient
	o.v[idx] = o.beta2*o.v[idx] + o.oneMinusBeta2*(gradient*gradient)

	return -o.lrt * (o.m[idx] / (math.Sqrt(o.v[idx]) + o.epsilon))
}

func fparam(val, fallback float64) float64 {
	if val == 0.0 {
		return fallback
	}
	return val
}

func iparam(val, fallback int) int {
	if val == 0 {
		return fallback
	}
	return val
}
