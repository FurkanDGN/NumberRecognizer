package me.github.furkandgn.nr.neuralnet.test;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * @author Furkan Doğan
 */
public class DL2Test {

  protected static int channels = 1;

  public static void main(String[] args) throws IOException {
    MultiLayerNetwork model = new MultiLayerNetwork(conf(540, 960, 3));
    model.init();

    File file = new File("image/line.jpeg");
    NativeImageLoader loader = new NativeImageLoader(540, 960, channels);
    INDArray indArray = loader.asMatrix(file);

    INDArray output = model.output(indArray);

    if (true) {
      System.out.println(output.slices());
      System.out.println(Arrays.toString(output.shape()));
      output = output.slice(0);
      System.out.println(output.slices());
      System.out.println(Arrays.toString(output.shape()));
      System.out.println(Arrays.toString(output.data().asDouble()));
      return;
    }

    for (int i = 0; i < 100; i++) {
      INDArray dup = Nd4j.create(output.dataType(), new long[]{1, 27, 48}, 'c');
      INDArray slice = output.slice(i);
      slice.mul(2);
      dup.putSlice(0, slice);

      resmiOlustur(dup, i + 1);
    }
  }

  public static MultiLayerConfiguration confa(int height, int width, int numLabels) {
    return new NeuralNetConfiguration.Builder()
      .l2(0.005)
      .activation(Activation.SIGMOID)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .list()
      .layer(0, convInit("conv1", channels, width, new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}, 0))
      .layer(1, new BatchNormalization.Builder().build())
      .layer(2, new ActivationLayer.Builder().activation(Activation.RELU).build())
      .layer(3, maxPool("max1", new int[]{2, 2}))


      .layer(4, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv2").nOut(width*2).build())
      .layer(5, new BatchNormalization.Builder().build())
      .layer(6, new ActivationLayer.Builder().activation(Activation.RELU).build())
      .layer(7, maxPool("max2", new int[]{2, 2}))

//      .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv3").nOut(128).build())
//      .layer(9, new BatchNormalization.Builder().build())
//      .layer(10, new ActivationLayer.Builder().activation(Activation.RELU).build())
//      .layer(11, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv4").nOut(128).build())
//      .layer(12, new DropoutLayer.Builder(0.35).build())
//      .layer(13, new BatchNormalization.Builder().build())
//      .layer(14, new ActivationLayer.Builder().activation(Activation.RELU).build())
//      .layer(15, maxPool("max3", new int[]{2, 2}))

//      .layer(16, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv5").nOut(512).build())
//      .layer(17, new BatchNormalization.Builder().build())
//      .layer(18, new ActivationLayer.Builder().activation(Activation.RELU).build())
//      .layer(19, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv6").nOut(512).build())
//      .layer(20, new DropoutLayer.Builder(0.35).build())
//      .layer(21, new BatchNormalization.Builder().build())
//      .layer(22, new ActivationLayer.Builder().activation(Activation.RELU).build())
//      .layer(23, maxPool("max4", new int[]{2, 2}))

      .layer(8, new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{0, 0}).name("conv7").nOut(width*4).build())
      .layer(9, new DropoutLayer.Builder(0.25).build())
      .layer(10, new BatchNormalization.Builder().build())
      .layer(11, new ActivationLayer.Builder().activation(Activation.RELU).build())

      .layer(12, new DenseLayer.Builder().nOut(width).build())
      .layer(13, new ActivationLayer.Builder().activation(Activation.SOFTMAX).build())
      .layer(14, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
      .nOut(numLabels)
      .build())
      .setInputType(InputType.convolutional(height, width, channels))
      .build();
  }

  public static MultiLayerConfiguration conf(int height, int width, int numLabels) {
    return new NeuralNetConfiguration.Builder()
      .updater(new Nesterovs (0.0015, 0.98))
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

  private static ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
    return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
  }

  private static ConvolutionLayer conv5x5(String name, int out, int[] stride, int[] pad, double bias) {
    return new ConvolutionLayer.Builder(new int[]{5, 5}, stride, pad).name(name).nOut(out).biasInit(bias).build();
  }

  private static SubsamplingLayer maxPool(String name, int[] kernel) {
    return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
  }

  private static void resmiOlustur(INDArray array, int id) {
    NativeImageLoader loader = new NativeImageLoader();
    Mat mat = loader.asMat(array);
    opencv_imgcodecs.imwrite(String.format("image/rendered/output-%s.jpg", id), mat);
  }
}
