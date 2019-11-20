var NN = require('./nn');
var neuron = require('./neuron');
var nn = new NN.NeuralNet();

const layers = [
  [neuron.neuron(2), neuron.neuron(2)],
  [neuron.neuron(2), neuron.neuron(2)],
  [neuron.neuron(2)],
];

var inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];

var outputs = [
  0,
  0,
  0,
  1
];

nn.setLayers(layers);
nn.setInputs(inputs);
nn.setOutputs(outputs);

var iter = 20;
while (iter--) {
  nn.forward_propagate(inputs[0], 0);
  var derivatives = nn.get_derivatives_wrt_activations(outputs[0]); 

  var output_neuron = nn.layers[nn.layers.length - 1][0];

  var old_activation = output_neuron.activation;

  var delA = 0.000001;
  nn.layers[0][0].activation += delA;
  nn.forward_propagate(nn.get_layer_activations(0), 1);
  var old_err = (old_activation - outputs[0]) * (old_activation - outputs[0])
  var new_err = (output_neuron.activation - outputs[0]) * (output_neuron.activation - outputs[0]);
  var delErr = new_err - old_err;
  nn.back_propagate(inputs[0], outputs[0]);
  console.log('obtained derivative', delErr / delA, 'computed derivative', derivatives[0][0]);
}
