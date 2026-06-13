package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.Map;

public class ForwardResult {
    public INDArray output;
    public Map<String, Object> cache;

    public ForwardResult(INDArray output, Map<String, Object> cache) {
        this.output = output;
        this.cache = cache;
    }
}