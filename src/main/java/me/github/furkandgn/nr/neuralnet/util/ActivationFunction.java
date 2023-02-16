package me.github.furkandgn.nr.neuralnet.util;

import java.util.function.Function;

public record ActivationFunction(Function<Double, Double> func, Function<Double, Double> dFunc) {

  public static final ActivationFunction SIGMOID = new ActivationFunction(
    x -> 1 / (1 + Math.exp(-x)),
    y -> y * (1 - y)
  );

  public static final ActivationFunction TANH = new ActivationFunction(
    Math::tanh,
    y -> 1 - (y * y)
  );

  public static final ActivationFunction RELU = new ActivationFunction(
    x -> x > 0d ? x : 0d,
    x -> x > 0d ? 1d : 0d
  );
}