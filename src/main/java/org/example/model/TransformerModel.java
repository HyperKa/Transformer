package org.example.model;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.example.ParameterContainer;

import org.example.data.ConfigConstants;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TransformerModel implements ParameterContainer, Serializable {

    private EmbeddingTable embeddingTable;
    private ArrayList<EncoderBlock> encoderBlocks;
    private PositionalEncoding pe;
    private ClassificationHead classificationHead;


    public TransformerModel(int vocabSize, int embeddingDim, int numHeads, int hiddenDim, int numLayers, int numClasses) throws IllegalAccessException {

        this.embeddingTable = new EmbeddingTable(vocabSize, embeddingDim);
        this.pe = new PositionalEncoding(ConfigConstants.MAX_LENGTH, ConfigConstants.EMBEDDING_DIM);
        this.encoderBlocks = new ArrayList<>();
        this.classificationHead = new ClassificationHead(embeddingDim, numClasses);

        for (int i = 0; i < numLayers; i++) {
            encoderBlocks.add(new EncoderBlock(numHeads, embeddingDim, hiddenDim));
        }
    }

    public ForwardResult forward(int[] sequence, int[] mask) {
        Map<String, Object> fullCache = new HashMap<>();

        // System.out.println("Длина последовательности в forward в TransformerModel: " + sequence.length);
        // RealMatrix wordEmbeddings = this.embeddingTable.getEmbeddingsVector(sequence);
        ForwardResult wordEmbeddings = this.embeddingTable.forward(sequence);

        RealMatrix posEmbeddings = this.pe.addPositionalEncoding(wordEmbeddings.output);  // Чтобы вы понимали, я думал тут сложение два раза идет... Мда, а время всего 10 утра)

        fullCache.put("embedding_cache", wordEmbeddings.cache);

        RealMatrix current_X = posEmbeddings;
        List<Map<String, Object>> encoderCaches = new ArrayList<>();
        for (EncoderBlock block : this.encoderBlocks) {
            // current_X = (RealMatrix) block.forward(current_X, mask);
            ForwardResult blockResult = block.forward(current_X, mask);
            current_X = blockResult.output;
            encoderCaches.add(blockResult.cache);
        }
        fullCache.put("encoder_caches", encoderCaches);

        RealMatrix encoderOutput = current_X;

        // Агрегация CLS токена:
        RealMatrix clsTokenOutput = encoderOutput.getRowMatrix(0); // Размер [1, 64]
        ForwardResult finalResult = this.classificationHead.forward(clsTokenOutput);
        fullCache.put("classification_head_cache", finalResult.cache);


        return new ForwardResult(finalResult.output, fullCache);
        // return finalResult;
    }

    public void backward(RealMatrix grad_output, Map<String, Object> cache) {
        Map<String, Object> chCache = (Map<String, Object>) cache.get("classification_head_cache");
        RealMatrix g_cls_token = this.classificationHead.backward(grad_output, chCache);

        RealMatrix g_encoder_output = new Array2DRowRealMatrix(ConfigConstants.MAX_LENGTH, ConfigConstants.EMBEDDING_DIM);
        g_encoder_output.setRowMatrix(0, g_cls_token);

        List<Map<String, Object>> encoderCaches = (List<Map<String, Object>>) cache.get("encoder_caches");
        RealMatrix currentGrad = g_encoder_output;
        for (int i = this.encoderBlocks.size() - 1; i >= 0; i--) {
            // System.out.println("Обратный обход по EncoderBlock, итерация: " + i);
            EncoderBlock block = encoderBlocks.get(i);
            Map<String, Object> blockCache = encoderCaches.get(i);
            currentGrad = block.backward(currentGrad, blockCache);
        }

        // Positional Encoding
        // currentGrad = this.pe.backward(currentGrad);  <-- интуитивно происходит это, результат не меняется

        Map<String, Object> wordEmbeddings = (Map<String, Object>) cache.get("embedding_cache");
        this.embeddingTable.backward(currentGrad, wordEmbeddings);
    }


    public PositionalEncoding getPe() {  // Получение позиционной матрицы для его добавления к эмбеддингам
        return pe;
    }

    public EmbeddingTable getEmbeddingTable() {
        return embeddingTable;
    }

    public ArrayList<EncoderBlock> getEncoderBlocks() {
        return encoderBlocks;
    }


    public void save(String filePath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(filePath);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {

            oos.writeObject(this); // Сериализуем весь объект модели
            System.out.println("Модель успешно сохранена в " + filePath);
        }
    }

    public static TransformerModel load(String filePath) throws IOException, ClassNotFoundException {
        try (FileInputStream fis = new FileInputStream(filePath);
             ObjectInputStream ois = new ObjectInputStream(fis)) {

            TransformerModel model = (TransformerModel) ois.readObject();
            System.out.println("Модель успешно загружена из (transformer_model.ser) корня проекта: " + filePath);
            return model;
        }
    }


    @Override
    public ArrayList<RealMatrix> getParameters() {  // Берем параметры из классов (SelfAttention -> MultiHeadAttention, FeedForwardNetwork, LayerNormalization) -> из EncoderBlock и EmbeddingTable
        ArrayList<RealMatrix> allParams = new ArrayList<>();
        allParams.addAll(this.embeddingTable.getParameters());
        for (EncoderBlock block : this.encoderBlocks) {
            allParams.addAll(block.getParameters());
        }
        allParams.addAll(this.classificationHead.getParameters());
        return allParams;
    }

    @Override
    public List<RealMatrix> getGradients() { // Тип возвращаемого значения List, а не RealMatrix
        List<RealMatrix> allGrads = new ArrayList<>();
        allGrads.addAll(this.embeddingTable.getGradients());
        for (EncoderBlock block : this.encoderBlocks) {
            allGrads.addAll(block.getGradients());
        }
        allGrads.addAll(this.classificationHead.getGradients());
        return allGrads;
    }

    @Override
    public void zeroGradients() {
        embeddingTable.zeroGradients();
        for (EncoderBlock block : encoderBlocks) {
            block.zeroGradients();
        }
        classificationHead.zeroGradients();
    }
}
