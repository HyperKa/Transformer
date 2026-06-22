package org.example.analysis;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.data.ConfigConstants;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.model.ForwardResult;
import org.example.model.TransformerModel;
import org.example.migration.ModelWeightLoader;

import java.io.IOException;
import java.util.List;

public class CodeAnalyzer {
    private final TransformerModel model;
    private final Vocabulary vocabulary;

    public CodeAnalyzer(String exportedModelDir) throws IOException, IllegalAccessException {
        System.out.println("=== Инициализация гибридного анализатора ===");

        this.vocabulary = Vocabulary.loadFromJson(exportedModelDir + "/vocab.json");

        this.model = new TransformerModel(
                vocabulary.getDictionarySize(),
                ConfigConstants.EMBEDDING_DIM,
                ConfigConstants.NUM_HEADS,
                ConfigConstants.FFN_HIDDEN_DIM,
                ConfigConstants.NUM_LAYERS,
                ConfigConstants.NUM_CLASSES
        );

        ModelWeightLoader.loadWeights(this.model, exportedModelDir);
        System.out.println("=== Инициализация успешно завершена! Модель готова к работе. ===\n");
    }

    public int analyze(String codeSnippet) {
        return analyzeDetailed(codeSnippet).predictedClass;
    }

    public AnalysisResult analyzeDetailed(String codeSnippet) {
        List<String> tokens = TrainingData.tokenizeCode(codeSnippet);
        int[][] resultParse = vocabulary.convertToIndexesForAnalysis(tokens);

        int[] sequence = resultParse[0];
        int[] mask = resultParse[1];

        System.out.printf("   [Токены] Всего обработано: %d | Известно модели: %d | Неизвестно (UNK): %d\n",
                tokens.size(), tokens.size() - getUnkCount(sequence), getUnkCount(sequence));

        ForwardResult result = model.forward(sequence, null);

        org.nd4j.linalg.api.ndarray.INDArray probabilities = org.nd4j.linalg.ops.transforms.Transforms.softmax(result.output);

        int predictedClass = org.nd4j.linalg.factory.Nd4j.argMax(probabilities, 1).getInt(0);
        double confidence = probabilities.getDouble(0, predictedClass);

        return new AnalysisResult(predictedClass, confidence);
    }

    // Вспомогательный метод для лога
    private int getUnkCount(int[] sequence) {
        int unk = 0;
        for (int i = 1; i < sequence.length; i++) {
            if (sequence[i] == 3) break; // EOS
            if (sequence[i] == 1) unk++;
        }
        return unk;
    }

    public static class AnalysisResult {
        public final int predictedClass;
        public final double confidence;

        public AnalysisResult(int predictedClass, double confidence) {
            this.predictedClass = predictedClass;
            this.confidence = confidence;
        }
    }
}