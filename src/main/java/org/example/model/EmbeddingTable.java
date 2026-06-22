package org.example.model;

import lombok.Setter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.example.repo.ParameterContainer;
import java.io.Serializable;
import java.util.*;

public class EmbeddingTable implements ParameterContainer, Serializable {
    private int size;
    private int embeddingDim;
    @Setter
    private INDArray table;
    private INDArray dTable;

    public EmbeddingTable(int size, int embeddingDim) {
        this.size = size;
        this.embeddingDim = embeddingDim;
        // Случайная инициализация [-0.1, 0.1]
        this.table = Nd4j.rand(new int[]{size, embeddingDim}).muli(0.2).subi(0.1);
        this.dTable = Nd4j.zeros(size, embeddingDim);
    }

    public ForwardResult forward(int[] sequence) {
        // В ND4J извлечение строк (lookup) делается через pullRows или интервалы
        INDArray embeddings = Nd4j.create(sequence.length, embeddingDim);
        for (int i = 0; i < sequence.length; i++) {
            int idx = (sequence[i] >= 0 && sequence[i] < size) ? sequence[i] : 1; // UNK = 1
            embeddings.putRow(i, table.getRow(idx));
        }

        Map<String, Object> cache = new HashMap<>();
        cache.put("sequence", sequence);
        return new ForwardResult(embeddings, cache);
    }

    public void backward(INDArray grad_output, Map<String, Object> cache) {
        int[] sequence = (int[]) cache.get("sequence");
        for (int i = 0; i < sequence.length; i++) {
            int idx = (sequence[i] >= 0 && sequence[i] < size) ? sequence[i] : 1;
            // Аккумулируем градиент для конкретного токена
            dTable.getRow(idx).addi(grad_output.getRow(i));
        }
    }

    @Override public List<INDArray> getParameters() { return List.of(table); }
    @Override public List<INDArray> getGradients() { return List.of(dTable); }
    @Override public void zeroGradients() { dTable.assign(0); }
}