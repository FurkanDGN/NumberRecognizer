package me.github.furkandgn.nr.neuralnet.test;

import me.github.furkandgn.nr.neuralnet.util.NeuralNetwork;
import me.github.furkandgn.nr.neuralnet.util.TrainData;
import me.github.furkandgn.nr.neuralnet.util.ActivationFunction;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * @author Furkan Doğan
 */
public class AlphabetTrain {

  private static final char[] ALPHABET = new char[]{'A', 'B', 'C', 'Ç', 'D', 'E', 'F', 'G', 'Ğ', 'H', 'İ', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V', 'Y', 'Z'};
  private static final int IMAGE_SIZE = 24;

  public static void main(String[] args) throws IOException {
    String pathRaw = "image/alphabet/a/";
    File folder = new File(pathRaw);
    File[] files = Objects.requireNonNull(folder.listFiles());
    int length = files.length;

    List<TrainData> trainDatas = new ArrayList<>();

    for (int fileIndex = 0; fileIndex < length; fileIndex++) {
      File file = files[fileIndex];
      BufferedImage image = ImageIO.read(file);

      double[] pixels = new double[IMAGE_SIZE * IMAGE_SIZE];

      int pixelsIndex = 0;
      for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
          int rgb = image.getRGB(i, j);
          double v = averageValue(rgb);
          double value = v / (255d);
          pixels[pixelsIndex++] = value;
        }
      }

      // trainDatas.add(new TrainData(pixels, 'a'));
    }

    double[] output = new double[ALPHABET.length];
    output[0] = 1;
    for (int i = 1; i < ALPHABET.length; i++) {
      output[i] = 0;
    }

    NeuralNetwork neuralNetwork = new NeuralNetwork(IMAGE_SIZE * IMAGE_SIZE, IMAGE_SIZE, ALPHABET.length, ActivationFunction.SIGMOID);
    neuralNetwork.randomize();

    for (int i = 0; i < 35; i++) {
      for (TrainData trainData : trainDatas) {
        neuralNetwork.train(trainData.pixels()[0], output);
      }

      for (TrainData trainData : trainDatas) {
        double[] predict = neuralNetwork.predict(trainData.pixels()[0]);
        System.out.println(predict[0]);
      }
    }
  }

  private static double averageValue(int rgb) {
    int r = (rgb >> 16) & 0xff;
    int g = (rgb >> 8) & 0xff;
    int b = rgb & 0xff;

    return (r + g + b) / 3d;
  }
}
