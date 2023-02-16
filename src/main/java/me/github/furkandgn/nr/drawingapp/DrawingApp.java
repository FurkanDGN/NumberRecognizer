package me.github.furkandgn.nr.drawingapp;

import me.github.furkandgn.nr.neuralnet.NetworkModels;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.utilty.SingletonDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * @author Furkan Doğan
 */
public class DrawingApp {

  private static final Logger logger = LoggerFactory.getLogger(DrawingApp.class);
  private static final int NETWORK_SIZE = 28;
  private static final int WIDTH = 700;
  private static final int HEIGHT = 500;

  static {
    nu.pattern.OpenCV.loadLocally();
  }

  public static void main(String[] args) throws IOException {
    DrawingApp drawingApp = new DrawingApp();
    drawingApp.startApp();
  }

  private void startApp() throws IOException {
    MultiLayerNetwork model = this.trainModel();

    AppWindow appWindow = new DefaultAppWindow("Text App", WIDTH, HEIGHT);
    appWindow.init();

    JLabel jLabel = this.addJLabel(appWindow);
    this.startPredictor(jLabel, appWindow, model);
  }

  private MultiLayerNetwork trainModel() throws IOException {
    double bestAccuracy = 0d;
    MultiLayerNetwork bestModel = null;

    DataSetIterator train = new MnistDataSetIterator(Integer.MAX_VALUE, 30000, true, true, true, 1023L);
    DataSet dataSet = train.next();
    KFoldIterator kFoldIterator = new KFoldIterator(5, dataSet);

    while (kFoldIterator.hasNext()) {
      DataSet next = kFoldIterator.next();
      MultiLayerNetwork model = new MultiLayerNetwork(NetworkModels.conf2(NETWORK_SIZE, NETWORK_SIZE, 10));
      model.init();

      model.fit(new SingletonDataSetIterator(next), 10);

      SingletonDataSetIterator testIterator = new SingletonDataSetIterator(kFoldIterator.testFold());
      Evaluation evaluate = model.evaluate(testIterator);
      double accuracy = evaluate.accuracy();
      System.out.println(accuracy);
      if (accuracy > bestAccuracy) {
        bestModel = model;
        bestAccuracy = accuracy;
      }
    }

    return bestModel;
  }

  private void startPredictor(JLabel jLabel, AppWindow appWindow, MultiLayerNetwork neuralNetwork) {
    NativeImageLoader loader = new NativeImageLoader(28, 28, 1);
    ScheduledExecutorService executorService = Executors.newScheduledThreadPool(1);
    executorService.scheduleAtFixedRate(() -> {
      Mat image = appWindow.image(NETWORK_SIZE, NETWORK_SIZE);

      INDArray pixelsArray;
      try {
        pixelsArray = loader.asMatrix(image);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }

      INDArray outputArray = neuralNetwork.output(pixelsArray);

      double[] output = outputArray.data().asDouble();

      double[] largest = this.largest(output);
      jLabel.setText(largest[0] + " " + largest[1]);
    }, 0, 500, TimeUnit.MILLISECONDS);
  }

  private double averageValue(int rgb) {
    int r = (rgb >> 16) & 0xff;
    int g = (rgb >> 8) & 0xff;
    int b = rgb & 0xff;

    return (r + g + b) / 3d;
  }

  private JLabel addJLabel(AppWindow appWindow) {
    JLabel jLabel = new JLabel(" ", SwingConstants.CENTER);
    jLabel.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 25));
    jLabel.setForeground(Color.BLACK);

    appWindow.addComponent(jLabel, gbc -> {
      gbc.fill = GridBagConstraints.CENTER;
      gbc.weightx = 0.0;
      gbc.gridx = 0;
      gbc.gridy++;
      gbc.gridwidth = 2;
      gbc.insets = new Insets(5, 0, 10, 0);
    });

    return jLabel;
  }

  double[] largest(double[] arr) {
    double max = Double.MIN_VALUE;
    int index = 0;

    for (int i = 0; i < arr.length; i++) {
      if (arr[i] > max) {
        max = arr[i];
        index = i;
      }
    }

    return new double[]{index, max};
  }

  private double[][] convertImage(BufferedImage image) {
    double[][] pixels = new double[NETWORK_SIZE][NETWORK_SIZE];

    for (int i = 0; i < NETWORK_SIZE; i++) {
      for (int j = 0; j < NETWORK_SIZE; j++) {
        int rgb = image.getRGB(i, j);
        double v = this.averageValue(rgb) / 255;
        pixels[i][j] = v;
      }
    }

    return pixels;
  }
}
