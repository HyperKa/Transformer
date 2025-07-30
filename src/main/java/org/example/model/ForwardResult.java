package org.example.model;

import org.apache.commons.math3.linear.RealMatrix;

import java.util.Map;

public class ForwardResult {

    public RealMatrix output;
    public Map<String, Object> cache;


    public ForwardResult(RealMatrix output, Map<String, Object> cache) {
        this.output = output;
        this.cache = cache;
    }
}
