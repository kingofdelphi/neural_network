const sigmoid = (x) => {
  return 1 / (1 + Math.exp(-x));
};

const dot = (a, b) => {
  let sum = 0;
  for (let i = 0; i < a.length; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
};

const neuron = (input_size) => ({
  inputs: new Array(input_size).fill(0),
  weights: new Array(input_size).fill(0).map(d => Math.random()),
  bias: 0,
  activation: 0,
  calculateActivation: function() {
    var result = sigmoid(dot(this.inputs, this.weights) + this.bias);
    return result;
  },
  activate: function() {
    this.activation = this.calculateActivation();
  },
  derivate_sigmoid_wrt_fsum: function() {
    var act = this.activation;
    return act * (1 - act);
  },
  derivate_fsum_wrt_inputs: function() {
    return this.weights;
  },
  derivate_fsum_wrt_weights: function() {
    return this.inputs;
  }
});


exports.neuron = neuron;
