package me.github.furkandgn.nr.neuralnet.util;

import java.awt.image.BufferedImage;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author Furkan Doğan
 */
public class AlphabetDataSet {

  public static List<TrainData> trainData() {
    List<TrainData> trainDatas = new ArrayList<>();

    String inputImagePath = "train-images-idx3-ubyte";
    String inputLabelPath = "train-labels-idx1-ubyte";

    int[] hashMap = new int[10];

    try (FileInputStream inImage = new FileInputStream(inputImagePath);
         FileInputStream inLabel = new FileInputStream(inputLabelPath)) {

      int magicNumberImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
      int numberOfImages = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
      int numberOfRows  = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());
      int numberOfColumns = (inImage.read() << 24) | (inImage.read() << 16) | (inImage.read() << 8) | (inImage.read());

      int magicNumberLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());
      int numberOfLabels = (inLabel.read() << 24) | (inLabel.read() << 16) | (inLabel.read() << 8) | (inLabel.read());

      BufferedImage image = new BufferedImage(numberOfColumns, numberOfRows, BufferedImage.TYPE_INT_ARGB);
      int numberOfPixels = numberOfRows * numberOfColumns;
      int[] imgPixels = new int[numberOfPixels];

      for(int i = 0; i < numberOfImages; i++) {

        int label = inLabel.read();

        if (label > 2) continue;

        if(i % 100 == 0) {System.out.println("Number of images extracted: " + i);}

        double[][] pixels = new double[28][28];

        for(int p = 0; p < numberOfPixels; p+=2) {
          int x = p % 28;
          int y = p / 28;
          int gray = inImage.read();
          imgPixels[p] = 0xFF000000 | (gray<<16) | (gray<<8) | gray;
          pixels[x][y] = averageValue(imgPixels[p]);
        }

        // hashMap[label]++;

        // image.setRGB(0, 0, numberOfColumns, numberOfRows, imgPixels, 0, numberOfColumns);
        // File outputfile = new File("image/mnist/" + label + "_0" + hashMap[label] + ".png");

        // ImageIO.write(image, "png", outputfile);

        trainDatas.add(new TrainData(pixels, String.valueOf(label).charAt(0)));
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

    return trainDatas;
  }

  private static double averageValue(int rgb) {
    int r = (rgb >> 16) & 0xff;
    int g = (rgb >> 8) & 0xff;
    int b = rgb & 0xff;

    return (r + g + b) / 3d;
  }
}
