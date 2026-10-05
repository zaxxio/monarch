package org.monarch.nn.data;

import java.util.ArrayList;
import java.util.List;

public class MLDataSet {
    private final List<MLData> dataList = new ArrayList<>();

    public MLDataSet(double[][] inputs, double[][] targets) {
        for (int i = 0; i < inputs.length; i++)
            dataList.add(new MLData(inputs[i], targets[i]));
    }
    public List<MLData> getDataList() { return dataList; }
}