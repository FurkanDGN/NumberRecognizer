package me.github.furkandgn.nr.neuralnet.test;

import me.github.furkandgn.nr.neuralnet.util.NeuralNetwork;
import me.github.furkandgn.nr.neuralnet.util.ActivationFunction;

/**
 * @author Furkan Doğan
 */
public class AndLogicTrain {

  static double [][] X= {
    {0,0},
    {1,0},
    {0,1},
    {1,1}
  };
  static double [][] Y= {
    {1},{0},{0},{1}
  };

  public static void main(String[] args) {

    NeuralNetwork nn = new NeuralNetwork(2,10,1, ActivationFunction.RELU);

    double[] output;

    nn.fit(X, Y, 50000);

    double [][] input = {
      {0,0},{1,0},{0,1},{1,1}
    };

    for(double[] d : input) {
      output = nn.predict(d);
      System.out.println(output[0] > 0.8);
    }
  }

  static double[] largest(double[] arr) {
    double max = arr[0];
    int maxIndex = 0;

    for (int i = 1; i < arr.length; i++)
      if (arr[i] > max) {
        max = arr[i];
        maxIndex = i;
        break;
      }

    return new double[]{maxIndex, max};
  }
}
