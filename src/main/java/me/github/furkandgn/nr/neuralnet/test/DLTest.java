package me.github.furkandgn.nr.neuralnet.test;

import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.Mat;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Adam;

import java.io.File;
import java.io.IOException;

/**
 * @author Furkan Doğan
 */
public class DLTest {

  public static void main(String[] args) throws IOException {
    for (int i = 0; i < 10; i++) {
      ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
        .updater(new Adam())
        .graphBuilder()
        .addInputs("model_input");

      builder.addLayer("conv1", new ConvolutionLayer.Builder()
        .nIn(1)
        .nOut(5)
        .kernelSize(3, 3)
        .padding(2, 2)
        .name("conv1")
        .weightInit(WeightInit.RELU)
        .build(), "model_input");
      builder.addLayer("pooling1", new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2,2)
        .stride(2,2)
        .build(), "conv1");
      builder.addLayer("act1", new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .name("act1")
        .build(), "pooling1");

      builder.setOutputs("act1");

      ComputationGraph model = new ComputationGraph(builder.build());
      model.init();

      File file = new File("image/line.jpeg");
      NativeImageLoader loader = new NativeImageLoader(540, 960, 1);
      INDArray indArray = loader.asMatrix(file);

      INDArray[] output = model.output(indArray);
      System.out.println(output.length);
      INDArray anan = output[0].mul(100);
      resmiOlustur(anan, i);
    }
    // INDArray[] indArrays =  model.output(indArray);
  }

  private static void resmiOlustur(INDArray array, int id) {
    NativeImageLoader loader = new NativeImageLoader();
    Mat mat = loader.asMat(array);
    opencv_imgcodecs.imwrite(String.format("output-%s.jpg", id), mat);
  }
}
