package me.github.furkandgn.nr.drawingapp;

import com.trlogic.testarea.view.canvas.MyCanvas;
import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;

/**
 * @author Furkan Doğan
 */
public class DefaultAppWindow extends Frame implements AppWindow {

  private final int width;
  private final int height;
  private final GridBagConstraints c = new GridBagConstraints();
  private Canvas canvas;

  public DefaultAppWindow(String title, int width, int height) throws HeadlessException {
    super(title);
    this.width = width;
    this.height = height;
  }

  @Override
  public void init() {
    this.setLayout(new GridBagLayout());

    MyCanvas canvas = new MyCanvas(500, 500);
    this.canvas = canvas;

    this.c.fill = GridBagConstraints.CENTER;
    this.c.gridx = 0;
    this.c.gridy = 0;
    this.c.gridwidth = 2;
    this.add(canvas, this.c);
    this.addWindowListener(new WindowAdapter() {
      public void windowClosing(WindowEvent we) {
        System.exit(0);
      }
    });

    this.addButtons(canvas);

    this.setSize(this.width, this.height);
    this.setVisible(true);
    this.setResizable(false);
  }

  private void addButtons(Canvas canvas) {
    JButton reset = new JButton("Sıfırla");
    reset.setSize(100, 50);
    JButton save = new JButton("Kaydet");
    save.setSize(100, 50);

    reset.addMouseListener(new MouseAdapter() {
      @Override
      public void mousePressed(MouseEvent e) {
        canvas.reset();
      }
    });

    save.addMouseListener(new MouseAdapter() {

      int count = 0;

      @Override
      public void mousePressed(MouseEvent e) {
        String pathRaw = String.format("image/alphabet/b/%s.jpg", this.count++);
        File file = new File(pathRaw);
        file.delete();
        try {
          Files.createDirectories(Path.of(pathRaw));
        } catch (IOException ex) {
          throw new RuntimeException(ex);
        }
        canvas.save(file);
      }
    });

    this.c.insets = new Insets(15, 0, 0, 0);
    this.c.fill = GridBagConstraints.HORIZONTAL;
    this.c.gridwidth = 1;
    this.c.weightx = 0.1;
    this.c.gridx = 0;
    this.c.gridy = 1;
    this.add(reset, this.c);
    this.c.gridx = 1;
    this.c.gridy = 1;
    this.add(save, this.c);
  }

  @Override
  public void addComponent(Object component, Consumer<GridBagConstraints> gridBagConstraintsConsumer) {
    if (!(component instanceof Component)) {
      throw new IllegalArgumentException("Component not supported");
    }

    gridBagConstraintsConsumer.accept(this.c);

    this.add((Component) component, this.c);
    this.pack();
    this.repaint();
  }

  @Override
  public Mat image(int width, int height) {
    return this.canvas.image(width, height);
  }
}
