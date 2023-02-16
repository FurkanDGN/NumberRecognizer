package me.github.furkandgn.nr.neuralnet.util;

import java.util.Arrays;
import java.util.function.Function;

/**
 * @author Furkan Doğan
 */
public class MatrixOld {

  private final int rows;
  private final int cols;

  private double[][] data;

  public MatrixOld(int rows, int cols) {
    this.rows = rows;
    this.cols = cols;
    this.data = new double[rows][cols];
    this.fill(0.0d);
  }

  public int getRows() {
    return this.rows;
  }

  public int getCols() {
    return this.cols;
  }

  public static MatrixOld fromArray(double[][] arr) {
    MatrixOld m = new MatrixOld(arr.length, arr[0].length);
    m.data = arr;
    return m;
  }

  public static MatrixOld fromArray(double[] arr) {
    MatrixOld m = new MatrixOld(arr.length, 1);
    for (int i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  public static MatrixOld subtract(MatrixOld a, MatrixOld b) {
    if (a.rows != b.rows || a.cols != b.cols) {
      throw new IllegalArgumentException("Matrices must have the same dimensions to substract them.");
    }

    MatrixOld result = new MatrixOld(a.rows, a.cols);

    for (int i = 0; i < a.data.length; i++) {
      for (int j = 0; j < a.data[i].length; j++) {
        result.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }

    return result;
  }

  public static MatrixOld transpose(MatrixOld m) {
    MatrixOld result = new MatrixOld(m.data[0].length, m.data.length);

    for (int i = 0; i < m.data.length; i++) {
      for (int j = 0; j < m.data[i].length; j++) {
        result.data[j][i] = m.data[i][j];
      }
    }

    return result;
  }

  public static MatrixOld multiply(MatrixOld a, MatrixOld b) {
    if (a.cols != b.rows) {
      throw new IllegalArgumentException("Matrices cannot be multiplied with these dimensions.");
    }

    MatrixOld result = new MatrixOld(a.rows, b.cols);

    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < b.cols; j++) {
        double sum = 0;
        for (int k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }

    return result;
  }

  public static MatrixOld map(MatrixOld m, Function<Double, Double> function) {
    MatrixOld result = new MatrixOld(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++) {
      for (int j = 0; j < m.cols; j++) {
        result.data[i][j] = function.apply(m.data[i][j]);
      }
    }

    return result;
  }

  public MatrixOld copy() {
    MatrixOld copy = new MatrixOld(this.rows, this.cols);
    for (int i = 0; i < this.rows; i++) {
      System.arraycopy(this.data[i], 0, copy.data[i], 0, this.cols);
    }
    return copy;
  }

  public double[] toArray() {
    double[] array = new double[this.rows * this.cols];

    int arrayIndex = 0;
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        array[arrayIndex++] = this.data[i][j];
      }
    }

    return array;
  }

  public void randomize() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double random = Math.random();
        this.data[i][j] = random;
      }
    }
  }

  public void add(MatrixOld other) {
    if (this.rows != other.rows || this.cols != other.cols) {
      throw new IllegalArgumentException("Matrices must have the same dimensions to add them.");
    }

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] += other.data[i][j];
      }
    }
  }

  public void multiply(MatrixOld other) {
    for(int i = 0; i < other.rows; i++) {
      for(int j = 0; j < other.cols; j++) {
        this.data[i][j] *= other.data[i][j];
      }
    }
  }

  public void multiply(double d) {
    this.map(e -> e + d);
  }

  public void map(Function<Double, Double> function) {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] = function.apply(this.data[i][j]);
      }
    }
  }

  public void print() {
    for (int i = 0; i < this.data.length; i++) {
      for (int j = 0; j < this.data[i].length; j++) {
        System.out.print(this.data[i][j] + "\t");
      }
      System.out.println();
    }
  }

  public void fill(double value) {
    for (double[] datum : this.data) {
      Arrays.fill(datum, value);
    }
  }
}
