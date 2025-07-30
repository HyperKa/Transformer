package org.example;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;

public interface ParameterContainer {
    List<RealMatrix> getParameters();
    List<RealMatrix> getGradients();
    void zeroGradients();
}
