package me.github.furkandgn.nr.neuralnet.util;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * @author Furkan Doğan
 */
public class Matrix {

  double[][] data;
  int rows, cols;

  public Matrix(int rows, int cols) {
    this.data = new double[rows][cols];
    this.rows = rows;
    this.cols = cols;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this.data[i][j] = Math.random() * 2 - 1;
      }
    }
  }

  public static Matrix fromArray(double[] x) {
    Matrix temp = new Matrix(x.length, 1);
    for (int i = 0; i < x.length; i++)
      temp.data[i][0] = x[i];
    return temp;

  }

  public static Matrix subtract(Matrix a, Matrix b) {
    Matrix temp = new Matrix(a.rows, a.cols);
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        temp.data[i][j] = a.data[i][j] - b.data[i][j];
      }
    }
    return temp;
  }

  public static Matrix transpose(Matrix a) {
    Matrix temp = new Matrix(a.cols, a.rows);
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        temp.data[j][i] = a.data[i][j];
      }
    }
    return temp;
  }

  public static Matrix multiply(Matrix a, Matrix b) {
    Matrix temp = new Matrix(a.rows, b.cols);
    for (int i = 0; i < temp.rows; i++) {
      for (int j = 0; j < temp.cols; j++) {
        double sum = 0;
        for (int k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        temp.data[i][j] = sum;
      }
    }
    return temp;
  }

  public void print() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        System.out.print(this.data[i][j] + " ");
      }
      System.out.println();
    }
  }

  public void add(int scaler) {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] += scaler;
      }

    }
  }

  public void add(Matrix m) {
    if (this.cols != m.cols || this.rows != m.rows) {
      System.out.println("Shape Mismatch");
      return;
    }

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] += m.data[i][j];
      }
    }
  }

  public double[] toArray() {
    List<Double> temp = new ArrayList<>();

    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        temp.add(this.data[i][j]);
      }
    }

    return temp.stream().mapToDouble(i -> i).toArray();
  }

  public void multiply(Matrix a) {
    for (int i = 0; i < a.rows; i++) {
      for (int j = 0; j < a.cols; j++) {
        this.data[i][j] *= a.data[i][j];
      }
    }

  }

  public void multiply(double a) {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] *= a;
      }
    }

  }

  public void sigmoid() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++)
        this.data[i][j] = 1 / (1 + Math.exp(-this.data[i][j]));
    }

  }

  public Matrix dsigmoid() {
    Matrix temp = new Matrix(this.rows, this.cols);
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++)
        temp.data[i][j] = this.data[i][j] * (1 - this.data[i][j]);
    }
    return temp;
  }

  public void map(Function<Double, Double> function) {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        this.data[i][j] = function.apply(this.data[i][j]);
      }
    }
  }

  public static Matrix map(Matrix m, Function<Double, Double> function) {
    Matrix result = new Matrix(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++) {
      for (int j = 0; j < m.cols; j++) {
        result.data[i][j] = function.apply(m.data[i][j]);
      }
    }

    return result;
  }

  public Matrix copy() {
    Matrix copy = new Matrix(this.rows, this.cols);
    for (int i = 0; i < this.rows; i++) {
      if (this.cols >= 0) System.arraycopy(this.data[i], 0, copy.data[i], 0, this.cols);
    }
    return copy;
  }

  public void randomize() {
    for (int i = 0; i < this.rows; i++) {
      for (int j = 0; j < this.cols; j++) {
        double random = Math.random();
        this.data[i][j] = random;
      }
    }
  }
}
