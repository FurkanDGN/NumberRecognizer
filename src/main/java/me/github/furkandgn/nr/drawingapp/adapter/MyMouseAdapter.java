package me.github.furkandgn.nr.drawingapp.adapter;

import com.trlogic.testarea.view.canvas.Canvas;

import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

/**
 * @author Furkan Doğan
 */
public class MyMouseAdapter extends MouseAdapter {

  private final Canvas canvas;
  private boolean pressed = false;
  private int startX;
  private int startY;

  public MyMouseAdapter(Canvas canvas) {
    this.canvas = canvas;
  }

  @Override
  public void mousePressed(MouseEvent e) {
    if (e.getButton() == MouseEvent.BUTTON1) {
      this.pressed = true;
      this.startX = e.getX();
      this.startY = e.getY();
    }
  }

  @Override
  public void mouseDragged(MouseEvent e) {
    if (e.getButton() == MouseEvent.BUTTON1 && this.pressed) {
      int endX = e.getX();
      int endY = e.getY();

      this.canvas.drawLine(this.startX, this.startY, endX, endY);

      this.startX = endX;
      this.startY = endY;
    }
  }

  @Override
  public void mouseReleased(MouseEvent e) {
    if (e.getButton() == MouseEvent.BUTTON1) {
      this.pressed = false;
      int endX = e.getX();
      int endY = e.getY();

      this.canvas.drawLine(this.startX, this.startY, endX, endY);
    }
  }


}
