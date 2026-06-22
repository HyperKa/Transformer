package org.example.parser.pipeline;

import org.example.parser.model.LLMVerdict;
import java.util.*;
import java.util.stream.Collectors;

public class MistralArbitrator {

    public static class AIResult {
        private final Map<Integer, Double> weights;
        private final List<Integer> rankedClasses;
        private final Map<String, Integer> headVerdicts;

        public AIResult(Map<Integer, Double> inputWeights, Map<String, Integer> headVerdicts) {
            this.weights = new HashMap<>();
            for (int i = 0; i <= 7; i++) this.weights.put(i, 0.0);
            this.weights.putAll(inputWeights);
            this.headVerdicts = headVerdicts;

            this.rankedClasses = this.weights.entrySet().stream()
                    .sorted((e1, e2) -> {
                        int weightCompare = Double.compare(e2.getValue(), e1.getValue());
                        if (weightCompare != 0) return weightCompare;
                        return e1.getKey().compareTo(e2.getKey()); // При равенстве приоритет меньшему классу (или SAFE)
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

        public int getSpecificHeadVerdict(String headName) {
            return headVerdicts.getOrDefault(headName, -1);
        }
    }

    public static AIResult getAnalysis(List<LLMVerdict> verdicts) {
        Map<Integer, Double> scores = new HashMap<>();
        Map<String, Integer> headVerdicts = new HashMap<>();

        for (LLMVerdict v : verdicts) {
            if (v.getPredictedClass() < 0) continue;

            headVerdicts.put(v.getHeadName(), v.getPredictedClass());

            double baseWeight = v.getConfidence() / 100.0;
            double adjustedWeight = calculateWeight(v.getHeadName(), v.getPredictedClass(), baseWeight);

            scores.merge(v.getPredictedClass(), adjustedWeight, Double::sum);
        }
        return new AIResult(scores, headVerdicts);
    }

    private static double calculateWeight(String headName, int predictedClass, double confidence) {
        double weight = confidence;

        // Узкоспециализированные бусты для пар экспертов
        if (headName.startsWith("H1_") && predictedClass == 1) weight *= 1.4;
        else if (headName.startsWith("H2_") && predictedClass == 2) weight *= 1.4;
        else if (headName.startsWith("H3_") && predictedClass == 3) weight *= 1.4;
        else if (headName.startsWith("H4_") && predictedClass == 4) weight *= 1.4;
        else if (headName.startsWith("H5_") && predictedClass == 5) weight *= 1.4;
        else if (headName.startsWith("H6_") && predictedClass == 6) weight *= 1.4;
        else if (headName.startsWith("H7_") && predictedClass == 7) weight *= 1.4;

        // Буст для архиватора
        if ("H17_ARCHITECT".equals(headName)) {
            weight *= 1.6;
        }

        return weight;
    }
}