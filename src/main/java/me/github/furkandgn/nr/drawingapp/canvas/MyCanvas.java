package me.github.furkandgn.nr.drawingapp.canvas;

import com.trlogic.testarea.view.adapter.MyMouseAdapter;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author Furkan Doğan
 */
public class MyCanvas extends AwtCanvas implements Canvas {

  private final Mat canvas;
  private final int width;
  private final int height;
  private final ImageIcon imageIcon;

  public MyCanvas(int width, int height) {
    this.width = width;
    this.height = height;
    this.canvas = new Mat(height, width, 16);
    this.imageIcon = new ImageIcon();

    this.reset();

    this.setSize(width, height);

    MyMouseAdapter myMouseAdapter = new MyMouseAdapter(this);

    this.addMouseMotionListener(myMouseAdapter);
    this.addMouseListener(myMouseAdapter);
  }

  @Override
  public void drawLine(int x1, int y1, int x2, int y2) {
    Imgproc.line(this.canvas, new Point(x1, y1), new Point(x2, y2), new Scalar(255, 255, 255), 30);
    this.imageIcon.setImage(HighGui.toBufferedImage(this.canvas));
    this.setIcon(this.imageIcon);
    this.repaint();
  }

  @Override
  public void reset() {
    Imgproc.rectangle(this.canvas, new Point(0, 0), new Point(this.width, this.height), new Scalar(0, 0, 0), -1);
    this.imageIcon.setImage(HighGui.toBufferedImage(this.canvas));
    this.setIcon(this.imageIcon);
    this.repaint();
  }

  @Override
  public void save(File file) {
    Mat output = new Mat(this.canvas.size(), this.canvas.type());
    Imgproc.resize(this.canvas, output, new Size(28, 28));
    BufferedImage image = (BufferedImage) HighGui.toBufferedImage(output);

    try {
      ImageIO.write(image, "png", file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Mat image(int width, int height) {
    Mat output = new Mat(this.canvas.size(), this.canvas.type());
    Imgproc.resize(this.canvas, output, new Size(width, height));
    return output;
  }
}
