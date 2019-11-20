const neuron = require('./neuron');

class NeuralNet {
  setLayers(layers) {
    this.layers = layers;
  }

  setInputs(inputs) {
    this.inputs = inputs;
  }

  setOutputs(outputs) {
    this.outputs = outputs;
  }

  forward_propagate(input, layer) {
    var prev_activations = input;
    for (let i = layer || 0; i < this.layers.length; ++i) {
      var activations = [];
      for (let j = 0; j < this.layers[i].length; ++j) {
        var neuron = this.layers[i][j];
        neuron.inputs = prev_activations;
        neuron.activate();
        activations.push(neuron.activation);
      }
      prev_activations = activations;
    }
  }

  get_layer_activations(layer_idx) {
    var activations = [];
    this.layers[layer_idx].forEach(neuron => {
      activations.push(neuron.activation);
    });
    return activations;
  }

  get_output_change_wrt_activation(layer_idx, neuron_idx, next_layer_derivatives) {
    var neuron_derivative = 0;
    for (var k = 0; k < this.layers[layer_idx + 1].length; ++k) { // for each neuron k in layer i + 1
      var neuron = this.layers[layer_idx + 1][k];
      neuron_derivative += neuron.weights[neuron_idx] * neuron.derivate_sigmoid_wrt_fsum() * next_layer_derivatives[k];
    }
    return neuron_derivative;
  }

  get_derivatives_wrt_activations(output) {
    var derivatives = this.layers.map(d => []);;
    var activation = this.layers[this.layers.length - 1][0].activation;
    // ((wx+b)-y)^2
    // dO/da*da/dI
    derivatives[derivatives.length - 1] = [2 * (activation - output)];
    for (var i = this.layers.length - 2; i >= 0; --i) { // layer i
      for (var j = 0; j < this.layers[i].length; ++j) { // for each neuron j in layer i
        var neuron_derivative = this.get_output_change_wrt_activation(i, j, derivatives[i + 1]);
        derivatives[i].push(neuron_derivative);
      }
    }
    return derivatives;
  }

  back_propagate(input, output) {
    // ((wx+b)-y)^2
    // dO/da*da/dI
    var derivatives = this.get_derivatives_wrt_activations(output);
    var learning_rate = 0.05;
    for (var i = this.layers.length - 1; i >= 0; --i) { // layer i
      for (var j = 0; j < this.layers[i].length; ++j) {
        var neuron = this.layers[i][j];
        var dsigmoid = neuron.derivate_sigmoid_wrt_fsum();
        for (var k = 0; k < neuron.weights.length; ++k) {
          var f = i == 0 ? input[k] : this.layers[i - 1][k].activation;
          neuron.weights[k] -= learning_rate * f * dsigmoid * derivatives[i][j];
        }
        neuron.bias -= learning_rate * dsigmoid * derivatives[i][j];
      }
    }
  }

  train(iter) {
    iter = iter || 200;
    while (iter--) {
      this.inputs.forEach((input, idx) => {
        this.forward_propagate(input);
        console.log(input, this.layers[this.layers.length - 1][0].activation);
        this.back_propagate(input, this.outputs[idx]);
      });
    }
  }

}

exports.NeuralNet = NeuralNet;
