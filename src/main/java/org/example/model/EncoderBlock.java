package org.example.model;

import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EncoderBlock implements ParameterContainer {  // Структура нейронной сети именно тут должна обретать целостность...
    private MultiHeadAttention mhaObject;
    private LayerNormalization norm1;
    private FeedForwardNetwork network;
    private LayerNormalization norm2;

    public EncoderBlock(int numHeads, int embeddingDim, int hiddenDim) throws IllegalAccessException {
        this.mhaObject = new MultiHeadAttention(embeddingDim, numHeads);
        this.norm1 = new LayerNormalization(embeddingDim, 1e-6);
        this.network = new FeedForwardNetwork(embeddingDim, hiddenDim);
        this.norm2 = new LayerNormalization(embeddingDim, 1e-6);
    }

    public ForwardResult forward(RealMatrix input, int[] attentionMask) {
        // --- Под-блок 1: Внимание + Add & Norm ---
        ForwardResult mhaResult = this.mhaObject.forward(input, attentionMask);
        RealMatrix partSum1 = input.add(mhaResult.output);  // Ко входной матрице эмбеддингов прибавляется матрица такой же размерности 100х64 из mha
        ForwardResult norm1Result = this.norm1.forward(partSum1);

        // --- Под-блок 2: FFN + Add & Norm ---
        ForwardResult ffnResult = this.network.forward(norm1Result.output);
        RealMatrix partSum2 = norm1Result.output.add(ffnResult.output);
        ForwardResult norm2Result = this.norm2.forward(partSum2);

        // --- Собираем кэш для всего блока ---
        Map<String, Object> cache = new HashMap<>();
        cache.put("input", input); // Нужен для Add & Norm
        cache.put("mha_cache", mhaResult.cache);
        cache.put("norm1_output", norm1Result.output); // Нужен для Add & Norm
        cache.put("norm1_cache", norm1Result.cache);
        cache.put("ffn_cache", ffnResult.cache);
        cache.put("norm2_cache", norm2Result.cache);

        return new ForwardResult(norm2Result.output, cache);
    }

    public RealMatrix backward(RealMatrix grad_output, Map<String, Object> cache) {
        RealMatrix input = (RealMatrix) cache.get("input");
        Map<String, Object> mha_cache = (Map<String, Object>) cache.get("mha_cache");
        RealMatrix norm1_output = (RealMatrix) cache.get("norm1_output");
        Map<String, Object> norm1_cache = (Map<String, Object>) cache.get("norm1_cache");
        Map<String, Object> ffn_cache = (Map<String, Object>) cache.get("ffn_cache");
        Map<String, Object> norm2_cache = (Map<String, Object>) cache.get("norm2_cache");

        // Все расчеты также в блокноте, копии градиентов объясняются там же математически

        // 1. Backward через norm2
        RealMatrix grad_partSum2 = this.norm2.backward(grad_output, norm2_cache);

        // 2. Backward через второе остаточное соединение (Add)
        // Градиент просто копируется на обе ветки
        RealMatrix grad_norm1_output = grad_partSum2.copy();
        RealMatrix grad_ffn_output = grad_partSum2.copy();

        // 3. Backward через FFN
        // Важно: grad_norm1_output нужно будет сложить с градиентом, который придет отсюда
        RealMatrix grad_from_ffn = this.network.backward(grad_ffn_output, ffn_cache);
        grad_norm1_output = grad_norm1_output.add(grad_from_ffn);

        // 4. Backward через norm1
        RealMatrix grad_partSum1 = this.norm1.backward(grad_norm1_output, norm1_cache);

        // 5. Backward через первое остаточное соединение (Add)
        RealMatrix grad_input_from_add = grad_partSum1.copy();
        RealMatrix grad_mha_output = grad_partSum1.copy();

        // 6. Backward через MHA
        RealMatrix grad_input_from_mha = this.mhaObject.backward(grad_mha_output, mha_cache);

        // 7. Суммируем градиенты по входу, так как он шел по двум путям
        RealMatrix final_grad_input = grad_input_from_add.add(grad_input_from_mha);

        return final_grad_input;
    }

    @Override
    public ArrayList<RealMatrix> getParameters() {
        ArrayList<RealMatrix> result = new ArrayList<>();
        result.addAll(mhaObject.getParameters());
        result.addAll(norm1.getParameters());
        result.addAll(network.getParameters());
        result.addAll(norm2.getParameters());
        return result;
    }

    @Override  // НЕ Пустышка, созданная для сохранения структуры ParameterContainer, запускать нужно backward проход сразу
    public List<RealMatrix> getGradients() {
        List<RealMatrix> result = new ArrayList<>();
        result.addAll(mhaObject.getGradients());
        result.addAll(norm1.getGradients());
        result.addAll(network.getGradients());
        result.addAll(norm2.getGradients());
        return result;
    }

    @Override  // аналогично ^^^
    public void zeroGradients() {
        this.mhaObject.zeroGradients();
        this.norm1.zeroGradients();
        this.network.zeroGradients();
        this.norm2.zeroGradients();
    }

}
