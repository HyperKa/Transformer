package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.example.repo.ParameterContainer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EncoderBlock implements ParameterContainer, Serializable {
    private MultiHeadAttention mhaObject;
    private LayerNormalization norm1;
    private FeedForwardNetwork network;
    private LayerNormalization norm2;

    public EncoderBlock(int numHeads, int embeddingDim, int hiddenDim) throws IllegalAccessException {
        this.mhaObject = new MultiHeadAttention(embeddingDim, numHeads);
        this.norm1 = new LayerNormalization(embeddingDim, 1e-5);
        this.network = new FeedForwardNetwork(embeddingDim, hiddenDim);
        this.norm2 = new LayerNormalization(embeddingDim, 1e-5);
    }

    public ForwardResult forward(INDArray input, int[] attentionMask) {
        // --- Под-блок 1: Внимание + Add & Norm ---
        ForwardResult mhaResult = this.mhaObject.forward(input, attentionMask);
        INDArray partSum1 = input.add(mhaResult.output); // Residual connection
        ForwardResult norm1Result = this.norm1.forward(partSum1);

        // --- Под-блок 2: FFN + Add & Norm ---
        ForwardResult ffnResult = this.network.forward(norm1Result.output);
        INDArray partSum2 = norm1Result.output.add(ffnResult.output); // Residual connection
        ForwardResult norm2Result = this.norm2.forward(partSum2);

        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input);
        cache.put("mha_cache", mhaResult.cache);
        cache.put("norm1_output", norm1Result.output);
        cache.put("norm1_cache", norm1Result.cache);
        cache.put("ffn_cache", ffnResult.cache);
        cache.put("norm2_cache", norm2Result.cache);

        return new ForwardResult(norm2Result.output, cache);
    }

    public INDArray backward(INDArray grad_output, Map<String, Object> cache) {
        Map<String, Object> mha_cache = (Map<String, Object>) cache.get("mha_cache");
        INDArray norm1_output = (INDArray) cache.get("norm1_output");
        Map<String, Object> norm1_cache = (Map<String, Object>) cache.get("norm1_cache");
        Map<String, Object> ffn_cache = (Map<String, Object>) cache.get("ffn_cache");
        Map<String, Object> norm2_cache = (Map<String, Object>) cache.get("norm2_cache");

        // 1. Backward через norm2
        INDArray grad_partSum2 = this.norm2.backward(grad_output, norm2_cache);

        // 2. Residual connection (Add) - градиент копируется
        INDArray grad_norm1_output = grad_partSum2.dup();
        INDArray grad_ffn_output = grad_partSum2.dup();

        // 3. Backward через FFN
        INDArray grad_from_ffn = this.network.backward(grad_ffn_output, ffn_cache);
        grad_norm1_output.addi(grad_from_ffn);

        // 4. Backward через norm1
        INDArray grad_partSum1 = this.norm1.backward(grad_norm1_output, norm1_cache);

        // 5. Residual connection (Add)
        INDArray grad_input_from_add = grad_partSum1.dup();
        INDArray grad_mha_output = grad_partSum1.dup();

        // 6. Backward через MHA
        INDArray grad_input_from_mha = this.mhaObject.backward(grad_mha_output, mha_cache);

        // 7. Итоговый градиент по входу
        return grad_input_from_add.addi(grad_input_from_mha);
    }

    public MultiHeadAttention getMha() { return mhaObject; }
    public LayerNormalization getNorm1() { return norm1; }
    public FeedForwardNetwork getNetwork() { return network; }
    public LayerNormalization getNorm2() { return norm2; }

    @Override
    public List<INDArray> getParameters() {
        List<INDArray> result = new ArrayList<>();
        result.addAll(mhaObject.getParameters());
        result.addAll(norm1.getParameters());
        result.addAll(network.getParameters());
        result.addAll(norm2.getParameters());
        return result;
    }

    @Override
    public List<INDArray> getGradients() {
        List<INDArray> result = new ArrayList<>();
        result.addAll(mhaObject.getGradients());
        result.addAll(norm1.getGradients());
        result.addAll(network.getGradients());
        result.addAll(norm2.getGradients());
        return result;
    }

    @Override
    public void zeroGradients() {
        this.mhaObject.zeroGradients();
        this.norm1.zeroGradients();
        this.network.zeroGradients();
        this.norm2.zeroGradients();
    }
}