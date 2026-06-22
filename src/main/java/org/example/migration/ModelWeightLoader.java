package org.example.migration;

import org.example.model.EncoderBlock;
import org.example.model.MultiHeadAttention;
import org.example.model.TransformerModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.io.File;

public class ModelWeightLoader {

    public static void loadWeights(TransformerModel model, String weightsDir) {
        // 1. Загрузка эмбеддингов
        INDArray emb = Nd4j.createFromNpyFile(new File(weightsDir + "/embedding_table.npy"));
        model.getEmbeddingTable().setTable(emb);

        // 2. Загрузка классификатора
        INDArray classW = Nd4j.createFromNpyFile(new File(weightsDir + "/classifier_w.npy"));
        INDArray classB = Nd4j.createFromNpyFile(new File(weightsDir + "/classifier_b.npy"));
        model.getClassificationHead().setWeights(classW, classB);

        // 3. Загрузка блоков трансформера
        int numLayers = model.getEncoderBlocks().size();
        for (int i = 0; i < numLayers; i++) {
            EncoderBlock block = model.getEncoderBlocks().get(i);
            String layerPath = weightsDir + "/layer_" + i;

            // Загрузка голов внимания (SelfAttention)
            MultiHeadAttention mha = block.getMha();
            int numHeads = mha.getHeads().size();
            for (int h = 0; h < numHeads; h++) {
                String headPath = layerPath + "/head_" + h;
                INDArray wq = Nd4j.createFromNpyFile(new File(headPath + "/Wq.npy"));
                INDArray wk = Nd4j.createFromNpyFile(new File(headPath + "/Wk.npy"));
                INDArray wv = Nd4j.createFromNpyFile(new File(headPath + "/Wv.npy"));
                mha.getHeads().get(h).setWeights(wq, wk, wv);
            }

            // Выходная проекция внимания
            INDArray wo = Nd4j.createFromNpyFile(new File(layerPath + "/Wo.npy"));
            mha.setWo(wo);

            // LayerNorm 1
            INDArray norm1Gamma = Nd4j.createFromNpyFile(new File(layerPath + "/norm1_gamma.npy"));
            INDArray norm1Beta = Nd4j.createFromNpyFile(new File(layerPath + "/norm1_beta.npy"));
            block.getNorm1().setParams(norm1Gamma, norm1Beta);

            // FFN
            INDArray ffnW1 = Nd4j.createFromNpyFile(new File(layerPath + "/ffn_w1.npy"));
            INDArray ffnB1 = Nd4j.createFromNpyFile(new File(layerPath + "/ffn_b1.npy"));
            INDArray ffnW2 = Nd4j.createFromNpyFile(new File(layerPath + "/ffn_w2.npy"));
            INDArray ffnB2 = Nd4j.createFromNpyFile(new File(layerPath + "/ffn_b2.npy"));
            block.getNetwork().setWeights(ffnW1, ffnB1, ffnW2, ffnB2);

            // LayerNorm 2
            INDArray norm2Gamma = Nd4j.createFromNpyFile(new File(layerPath + "/norm2_gamma.npy"));
            INDArray norm2Beta = Nd4j.createFromNpyFile(new File(layerPath + "/norm2_beta.npy"));
            block.getNorm2().setParams(norm2Gamma, norm2Beta);
        }
        System.out.println("Все веса успешно перенесены в модель ND4J.");
    }
}