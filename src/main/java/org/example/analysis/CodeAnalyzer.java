package org.example.analysis;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.example.data.TrainingData;
import org.example.data.Vocabulary;
import org.example.data.ConfigConstants;
import org.example.model.ForwardResult;
import org.example.model.TransformerModel;

import java.io.IOException;
import java.util.List;

public class CodeAnalyzer {
    private final TransformerModel model;
    private final Vocabulary vocabulary;

    public CodeAnalyzer(String modelPath, String vocabPath) throws IOException, ClassNotFoundException {
        this.model = TransformerModel.load(modelPath);
        this.vocabulary = Vocabulary.load(vocabPath);
    }

    public int analyze(String codeSnippet) {
        List<String> tokens = TrainingData.tokenizeCode(codeSnippet);
        int[][] resultParse = vocabulary.convertToIndexesForAnalysis(tokens);

        int[] sequence = resultParse[0];
        int[] mask = resultParse[1];

        ForwardResult result = model.forward(sequence, mask);

        // Используем встроенный Softmax ND4J
        INDArray probabilities = Transforms.softmax(result.output);

        // Получаем индекс максимального значения (предсказанный класс)
        int predictedClass = Nd4j.argMax(probabilities, 1).getInt(0);
        double confidence = probabilities.getDouble(0, predictedClass);

        System.out.printf("Анализ завершен. Предсказанный класс: %d с уверенностью %.2f%%\n",
                predictedClass, confidence * 100);
        return predictedClass;
    }
}