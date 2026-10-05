package org.monarch.nn.learningrate;

public interface LRSchedule {
    double getLR(int epoch, int step);
    String name();
}