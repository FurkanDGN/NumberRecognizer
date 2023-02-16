package me.github.furkandgn.nr.neuralnet;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author Furkan Doğan
 */
public class NetworkModels {

  protected static int channels = 1;

  public static MultiLayerConfiguration conf1(int height, int width, int numLabels) {
    return new NeuralNetConfiguration.Builder()
      .updater(new Nesterovs(0.0015, 0.98))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nIn(784)
        .nOut(1000)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nOut(10)
        .build())
      .setInputType(InputType.convolutional(height, width, channels))
      .build();
  }

  public static MultiLayerConfiguration conf2(int height, int width, int numLabels) {
    return new NeuralNetConfiguration.Builder()
      .updater(new Nesterovs(0.0015, 0.98))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(8).activation(Activation.RELU).build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
      .layer(2, new ConvolutionLayer.Builder(3,3).stride(1,1).nOut(16).activation(Activation.RELU).build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2,2).stride(2,2).build())
      .layer(4, new DenseLayer.Builder()
        .activation(Activation.RELU)
        .nOut(1000)
        .build())
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .nOut(numLabels)
        .build())
      .setInputType(InputType.convolutionalFlat(height, width, channels))
      .build();
  }
}
