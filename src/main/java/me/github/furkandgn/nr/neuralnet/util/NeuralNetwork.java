package me.github.furkandgn.nr.neuralnet.util;

import java.util.function.Function;

/**
 * @author Furkan Doğan
 */
public class NeuralNetwork {

  private static final double INITIAL_LEARNING_RATE = 0.1d;

  private final Matrix weightsIH;
  private final Matrix weightsHO;
  private final Matrix biasH;
  private final Matrix biasO;
  private double learningRate = INITIAL_LEARNING_RATE;
  private ActivationFunction activationFunction;

  public NeuralNetwork(NeuralNetwork neuralNetwork) {
    this.weightsIH = neuralNetwork.weightsIH.copy();
    this.weightsHO = neuralNetwork.weightsHO.copy();
    this.biasH = neuralNetwork.biasH.copy();
    this.biasO = neuralNetwork.biasO.copy();
    this.activationFunction = neuralNetwork.activationFunction;
  }

  public NeuralNetwork(int inputNodes,
                       int hiddenNodes,
                       int outputNodes) {
    this(inputNodes, hiddenNodes, outputNodes, ActivationFunction.RELU);
  }

  public NeuralNetwork(int inputNodes,
                       int hiddenNodes,
                       int outputNodes,
                       ActivationFunction activationFunction) {
    this.weightsIH = new Matrix(hiddenNodes, inputNodes);
    this.weightsHO = new Matrix(outputNodes, hiddenNodes);
    this.biasH = new Matrix(hiddenNodes, 1);
    this.biasO = new Matrix(outputNodes, 1);
    this.activationFunction = activationFunction;
  }

  public double[] predict(double[] input) {
    Matrix inputs = Matrix.fromArray(input);
    Matrix hidden = Matrix.multiply(this.weightsIH, inputs);
    hidden.add(this.biasH);
    hidden.map(this.activationFunction.func());

    Matrix output = Matrix.multiply(this.weightsHO, hidden);
    output.add(this.biasO);
    output.map(this.activationFunction.func());

    return output.toArray();
  }

  public void fit(double[][] X, double[][] Y, int epochs) {
    for(int i=0;i<epochs;i++) {
      int sampleN =  (int)(Math.random() * X.length );
      this.train(X[sampleN], Y[sampleN]);
    }
  }

  public void train(double[] x, double[] y) {
    Matrix inputs = Matrix.fromArray(x);
    Matrix hidden = Matrix.multiply(this.weightsIH, inputs);
    hidden.add(this.biasH);
    hidden.map(this.activationFunction.func());

    Matrix outputs = Matrix.multiply(this.weightsHO, hidden);
    outputs.add(this.biasO);
    outputs.map(this.activationFunction.func());

    Matrix targets = Matrix.fromArray(y);

    Matrix outputErrors = Matrix.subtract(targets, outputs);
    Matrix gradients = Matrix.map(outputs, this.activationFunction.dFunc());
    gradients.multiply(outputErrors);
    gradients.multiply(this.learningRate);

    Matrix hiddenT = Matrix.transpose(hidden);
    Matrix weightHODeltas = Matrix.multiply(gradients, hiddenT);

    this.weightsHO.add(weightHODeltas);
    this.biasO.add(gradients);

    Matrix whoT = Matrix.transpose(this.weightsHO);
    Matrix hiddenErrors = Matrix.multiply(whoT, outputErrors);

    Matrix hiddenGradient = Matrix.map(hidden, this.activationFunction.dFunc());
    hiddenGradient.multiply(hiddenErrors);
    hiddenGradient.multiply(this.learningRate);

    Matrix inputsT = Matrix.transpose(inputs);
    Matrix weightIHDeltas = Matrix.multiply(hiddenGradient, inputsT);

    this.weightsIH.add(weightIHDeltas);
    this.biasH.add(hiddenGradient);
  }

  public void randomize() {
    this.weightsIH.randomize();
    this.weightsHO.randomize();
    this.biasH.randomize();
    this.biasO.randomize();
  }

  public void mutate(Function<Double, Double> func) {
    this.weightsIH.map(func);
    this.weightsHO.map(func);
    this.biasH.map(func);
    this.biasO.map(func);
  }

  public NeuralNetwork copy() {
    return new NeuralNetwork(this);
  }

  private void setLearningRate(double learningRate) {
    this.learningRate = learningRate;
  }

  private void setActivationFunction(ActivationFunction activationFunction) {
    this.activationFunction = activationFunction;
  }
}
