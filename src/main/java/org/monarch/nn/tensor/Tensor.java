package org.monarch.nn.tensor;

import java.io.Serializable;
import java.util.Arrays;

public final class Tensor implements Serializable {

    private final int[] shape;
    private final double[] data;

    public Tensor(int... shape) {
        this.shape = shape.clone();
        int size = 1;
        for (int s : shape) size *= s;
        this.data = new double[size];
    }

    public Tensor(double[] data, int... shape) {
        this.shape = shape.clone();
        int expected = 1;
        for (int s : shape) expected *= s;
        if (data.length != expected)
            throw new IllegalArgumentException(
                "data length " + data.length + " != shape size " + expected);
        this.data = data;
    }

    public int[] getShape() { return shape.clone(); }
    public double[] getData() { return data; }
    public int rank() { return shape.length; }
    public int size() { return data.length; }
    public int dim(int i) { return shape[i]; }

    public double get(int... idx) { return data[flat(idx)]; }
    public void set(double v, int... idx) { data[flat(idx)] = v; }
    public void add(double v, int... idx) { data[flat(idx)] += v; }

    public void zero() { Arrays.fill(data, 0.0); }

    public Tensor copy() { return new Tensor(data.clone(), shape); }

    private int flat(int... idx) {
        int i = 0, stride = 1;
        for (int d = shape.length - 1; d >= 0; d--) {
            i += idx[d] * stride;
            stride *= shape[d];
        }
        return i;
    }

    @Override
    public String toString() { return "Tensor" + Arrays.toString(shape); }
}