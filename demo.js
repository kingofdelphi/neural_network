var NN = require('./nn');
var neuron = require('./neuron');

var nn = new NN.NeuralNet();

const layers = [
  [neuron.neuron(2), neuron.neuron(2)],
  [neuron.neuron(2), neuron.neuron(2)],
  [neuron.neuron(2)]
];

var inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

var outputs = [
  0,
  1,
  1,
  0
];

nn.setLayers(layers);
nn.setInputs(inputs);
nn.setOutputs(outputs);

nn.train(80000);
