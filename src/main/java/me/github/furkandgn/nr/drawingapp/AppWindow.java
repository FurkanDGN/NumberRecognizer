package me.github.furkandgn.nr.drawingapp;

import org.opencv.core.Mat;

import java.awt.*;
import java.util.function.Consumer;

/**
 * @author Furkan Doğan
 */
public interface AppWindow {

  void init();

  void addComponent(Object component, Consumer<GridBagConstraints> gridBagConstraintsConsumer);

  Mat image(int width, int height);
}
