package org.example.repo;

import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.List;

public interface ParameterContainer {
    List<INDArray> getParameters();
    List<INDArray> getGradients();
    void zeroGradients();
}