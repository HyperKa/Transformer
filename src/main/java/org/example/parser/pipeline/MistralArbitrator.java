package org.example.parser.pipeline;

import org.example.parser.model.LLMVerdict;
import java.util.*;
import java.util.stream.Collectors;

public class MistralArbitrator {

    public static class AIResult {
        private final Map<Integer, Double> weights;
        private final List<Integer> rankedClasses;

        public AIResult(Map<Integer, Double> inputWeights) {
            this.weights = new HashMap<>();
            for (int i = 0; i <= 7; i++) this.weights.put(i, 0.0);
            this.weights.putAll(inputWeights);

            this.rankedClasses = this.weights.entrySet().stream()
                    .sorted((e1, e2) -> {
                        // сравнение по весу, перестановка местами для сортировки по убыванию
                        int weightCompare = Double.compare(e2.getValue(), e1.getValue());

                        // классический выход для неравных весов
                        if (weightCompare != 0) return weightCompare;

                        // спорно, пока тут в приоритете 0 класс, если веса равны, в приоритете 0
                        return e1.getKey().compareTo(e2.getKey());
                    })
                    .map(Map.Entry::getKey)
                    .collect(Collectors.toUnmodifiableList());
        }

        public double getWeight(int clazz) {
            return weights.getOrDefault(clazz, 0.0);
        }

        public int getWinner() {
            return rankedClasses.get(0);
        }

        public List<Integer> getTopN(int n) {
            return rankedClasses.subList(0, Math.min(n, rankedClasses.size()));
        }

        public Map<Integer, Double> getAllWeights() {
            return Collections.unmodifiableMap(weights);
        }
    }

    public static AIResult getAnalysis(List<LLMVerdict> verdicts) {
        Map<Integer, Double> scores = new HashMap<>();
        for (LLMVerdict v : verdicts) {
            if (v.getPredictedClass() < 0) continue;
            double weight = calculateWeight(v.getHeadName(), v.getPredictedClass(), v.getConfidence() / 100.0);
            // метод для суммирования весов, если класс уже есть, то добавляем к существующему (формат key-value)
            scores.merge(v.getPredictedClass(), weight, Double::sum);
        }
        return new AIResult(scores);
    }

    private static double calculateWeight(String headName, int predictedClass, double confidence) {
        double weight = confidence;
        switch (headName) {
            case "H2_ARCHITECT": if (predictedClass == 0) weight *= 1.5; break;
            case "H1_STRICT": case "H3_HACKER": if (predictedClass > 0) weight *= 1.3; break;
            case "H4_RESOURCES": if (predictedClass == 1 || predictedClass == 4) weight *= 1.4; break;
            case "H6_EXCEPTIONS": if (predictedClass == 2) weight *= 1.4; break;
        }
        return weight;
    }
}