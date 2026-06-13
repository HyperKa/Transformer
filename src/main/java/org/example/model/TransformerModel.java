package org.example.model;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.example.repo.ParameterContainer;
import org.example.data.ConfigConstants;
import java.io.*;
import java.util.*;

public class TransformerModel implements ParameterContainer, Serializable {
    private EmbeddingTable embeddingTable;
    private List<EncoderBlock> encoderBlocks;
    private PositionalEncoding pe;
    private ClassificationHead classificationHead;
    private final int embDim;

    public TransformerModel(int vocabSize, int embDim, int heads, int hidden, int layers, int classes) throws IllegalAccessException {
        this.embDim = embDim;
        this.embeddingTable = new EmbeddingTable(vocabSize, embDim);
        this.pe = new PositionalEncoding(ConfigConstants.MAX_LENGTH, embDim);
        this.encoderBlocks = new ArrayList<>();
        this.classificationHead = new ClassificationHead(embDim, classes);

        for (int i = 0; i < layers; i++) {
            encoderBlocks.add(new EncoderBlock(heads, embDim, hidden));
        }
    }

    public ForwardResult forward(int[] sequence, int[] mask) {
        ForwardResult embRes = embeddingTable.forward(sequence);
        INDArray x = pe.addPositionalEncoding(embRes.output);

        List<Map<String, Object>> encCaches = new ArrayList<>();
        for (EncoderBlock block : encoderBlocks) {
            ForwardResult res = block.forward(x, mask);
            x = res.output;
            encCaches.add(res.cache);
        }

        // CLS токен (строка 0)
        INDArray clsToken = x.get(NDArrayIndex.point(0), NDArrayIndex.all()).reshape(1, embDim);
        ForwardResult finalRes = classificationHead.forward(clsToken);

        Map<String, Object> fullCache = new HashMap<>();
        fullCache.put("emb_cache", embRes.cache);
        fullCache.put("enc_caches", encCaches);
        fullCache.put("head_cache", finalRes.cache);
        return new ForwardResult(finalRes.output, fullCache);
    }

    public void backward(INDArray grad_output, Map<String, Object> cache) {
        Map<String, Object> headCache = (Map<String, Object>) cache.get("head_cache");
        INDArray gCls = classificationHead.backward(grad_output, headCache);

        INDArray gEncoder = Nd4j.zeros(ConfigConstants.MAX_LENGTH, embDim);
        gEncoder.putRow(0, gCls);

        List<Map<String, Object>> encCaches = (List<Map<String, Object>>) cache.get("enc_caches");
        for (int i = encoderBlocks.size() - 1; i >= 0; i--) {
            gEncoder = encoderBlocks.get(i).backward(gEncoder, encCaches.get(i));
        }

        embeddingTable.backward(gEncoder, (Map<String, Object>) cache.get("emb_cache"));
    }

    public static TransformerModel load(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path))) {
            return (TransformerModel) ois.readObject();
        }
    }

    public void save(String path) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
            oos.writeObject(this);
        }
    }

    @Override public List<INDArray> getParameters() {
        List<INDArray> p = new ArrayList<>(embeddingTable.getParameters());
        for (EncoderBlock b : encoderBlocks) p.addAll(b.getParameters());
        p.addAll(classificationHead.getParameters());
        return p;
    }

    @Override public List<INDArray> getGradients() {
        List<INDArray> g = new ArrayList<>(embeddingTable.getGradients());
        for (EncoderBlock b : encoderBlocks) g.addAll(b.getGradients());
        g.addAll(classificationHead.getGradients());
        return g;
    }

    @Override public void zeroGradients() {
        embeddingTable.zeroGradients();
        for (EncoderBlock b : encoderBlocks) b.zeroGradients();
        classificationHead.zeroGradients();
    }
}