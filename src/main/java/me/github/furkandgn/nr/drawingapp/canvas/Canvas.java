package me.github.furkandgn.nr.drawingapp.canvas;

import org.opencv.core.Mat;

import java.io.File;

/**
 * @author Furkan Doğan
 */
public interface Canvas {

  void reset();

  void save(File file);

  void drawLine(int x1, int y1, int x2, int y2);

  Mat image(int width, int height);
}
