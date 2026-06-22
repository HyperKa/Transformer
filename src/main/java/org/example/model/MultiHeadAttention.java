package org.example.model;

import lombok.Getter;
import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.example.repo.ParameterContainer;

import java.io.Serializable;
import java.util.*;

public class MultiHeadAttention implements ParameterContainer, Serializable {
    private int embeddingDim;
    private int numHeads;
    @Getter
    private List<SelfAttention> heads;
    @Setter
    private INDArray Wo;
    private INDArray dWo;

    public MultiHeadAttention(int embeddingDim, int numHeads) {
        this.embeddingDim = embeddingDim;
        this.numHeads = numHeads;
        int headDim = embeddingDim / numHeads;

        this.heads = new ArrayList<>();
        for (int i = 0; i < numHeads; i++) {
            heads.add(new SelfAttention(embeddingDim, headDim));
        }
        this.Wo = Nd4j.rand(embeddingDim, embeddingDim).subi(0.5).muli(0.2);
        this.dWo = Nd4j.zeros(embeddingDim, embeddingDim);
    }

    public ForwardResult forward(INDArray input, int[] mask) {
        INDArray[] headOutputs = new INDArray[numHeads];
        List<Map<String, Object>> headCaches = new ArrayList<>();

        for (int i = 0; i < numHeads; i++) {
            ForwardResult res = heads.get(i).forward(input, mask);
            headOutputs[i] = res.output;
            headCaches.add(res.cache);
        }

        // Склейка голов по горизонтали (dim 1)
        INDArray concatenated = Nd4j.concat(1, headOutputs);
        INDArray output = concatenated.mmul(Wo);

        Map<String, Object> cache = new HashMap<>();
        cache.put("concatenated", concatenated);
        cache.put("headCaches", headCaches);
        return new ForwardResult(output, cache);
    }

    public INDArray backward(INDArray gradOutput, Map<String, Object> cache) {
        INDArray concatenated = (INDArray) cache.get("concatenated");
        List<Map<String, Object>> headCaches = (List<Map<String, Object>>) cache.get("headCaches");

        this.dWo.addi(concatenated.transpose().mmul(gradOutput));
        INDArray dConcatenated = gradOutput.mmul(Wo.transpose());

        INDArray dxTotal = Nd4j.zeros(gradOutput.shape());
        int headDim = embeddingDim / numHeads;

        for (int i = 0; i < numHeads; i++) {
            INDArray headGrad = dConcatenated.get(NDArrayIndex.all(), NDArrayIndex.interval(i * headDim, (i + 1) * headDim));
            dxTotal.addi(heads.get(i).backward(headGrad, headCaches.get(i)));
        }
        return dxTotal;
    }

    @Override public List<INDArray> getParameters() {
        List<INDArray> params = new ArrayList<>(List.of(Wo));
        for (SelfAttention h : heads) params.addAll(h.getParameters());
        return params;
    }

    @Override public List<INDArray> getGradients() {
        List<INDArray> grads = new ArrayList<>(List.of(dWo));
        for (SelfAttention h : heads) grads.addAll(h.getGradients());
        return grads;
    }

    @Override public void zeroGradients() {
        dWo.assign(0);
        for (SelfAttention h : heads) h.zeroGradients();
    }
}